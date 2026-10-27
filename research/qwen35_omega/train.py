"""Matched single-GPU Stage A/B training with strict adapter checkpoints."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed
from .cache import CACHE, atomic_save, identity
from .runtime import ARMS, RUN, AlignedModel, add_lora, load_qwen, load_native_vision, collate, processor, read_rows, restore_trainable, trainable_state


@lru_cache(maxsize=128)
def feature(kind, digest):
    value = torch.load(CACHE / kind / (digest+".pt"), map_location="cpu", weights_only=True)
    if value["image_sha256"] != digest or value["cache_identity"] != identity():
        raise RuntimeError("Feature identity mismatch")
    return value


def make_batch(records, arm, training=True):
    ids, labels, mm, grids, native, auxiliary = [], [], [], [], [], []
    for r in records:
        n = r["input_ids"].numel() if training else r["prompt_length"]
        ids.append(r["input_ids"][:n])
        labels.append(r["labels"][:n])
        mm.append(r["mm_token_type_ids"][:n])
        raw = feature("native", r["image_sha256"])
        if not torch.equal(raw["grid"], r["grid"]):
            raise RuntimeError("Token and native feature grid mismatch")
        grids.append(r["grid"])
        native.append(raw["native"])
        if ARMS[arm]:
            auxiliary.append(feature(ARMS[arm], r["image_sha256"]))
    # A common 128-token bucket limits kernel recompilation across lengths.
    # Left padding remains fully masked and never truncates images or answers.
    length = math.ceil(max(v.numel() for v in ids)/128)*128
    def pad(values, fill):
        return torch.stack([torch.nn.functional.pad(v, (length-v.numel(), 0), value=fill) for v in values])
    batch = {"input_ids": pad(ids, 0), "attention_mask": pad([torch.ones_like(v) for v in ids], 0),
             "mm_token_type_ids": pad(mm, 0), "image_grid_thw": torch.cat(grids),
             "pixel_values": torch.empty(0), "labels": pad(labels, -100)}
    context = {"native": torch.cat(native).cuda(non_blocking=True), "grid": batch["image_grid_thw"].cuda()}
    if auxiliary:
        for key in ("local", "local_valid", "pooled_scene", "pooled_valid", "register_scene"):
            if key in auxiliary[0]:
                context[key] = torch.stack([a[key] for a in auxiliary]).cuda(non_blocking=True)
    return batch, context


def optimizer_for(model, stage):
    groups = {}
    rates = {"fusion": 1e-4 if stage == "A" else 5e-5, "merger": 1e-5, "lora": 2e-5}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "fusion" if name.startswith("spatial_alignment.") else "lora" if "lora_" in name else "merger"
        decay = p.ndim > 1 and not any(part in name for part in ("norm", "gate", "bias"))
        key = (kind, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": rates[kind], "initial_lr": rates[kind],
                           "weight_decay": 0.01 if decay else 0., "component": kind}
        groups[key]["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9,0.95), eps=1e-8)


def epoch_order(records, seed, epoch):
    rng = random.Random(seed+epoch*1009)
    order = list(range(len(records)))
    rng.shuffle(order)
    # Identical bounded length grouping for all arms; preserve each record once.
    for start in range(0,len(order),512):
        order[start:start+512] = sorted(order[start:start+512], key=lambda i: records[i]["input_ids"].numel())
    return order


def initialization_hashes(model):
    digests = {name:hashlib.sha256() for name in ("merger","lora","fusion")}
    for name,p in model.named_parameters():
        kind = "lora" if "lora_" in name else "fusion" if name.startswith("spatial_alignment.") else "merger" if ".visual.merger." in name else None
        if kind:
            digests[kind].update(name.encode())
            digests[kind].update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {name:d.hexdigest() for name,d in digests.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=[k for k in ARMS if k != "R0"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("train", "full_train"), default="train")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--short-microbatch",type=int,default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if 32 % args.microbatch:
        raise ValueError("Microbatch must divide effective batch 32")
    if args.short_microbatch and (args.microbatch,args.short_microbatch)!=(16,32):
        raise ValueError("Only the validated 16-long / 32-short policy is supported")
    for kind in ("tokens", "native", ARMS[args.arm]):
        if kind and not (CACHE / (kind+"_complete.json")).exists():
            raise RuntimeError(f"Incomplete cache: {kind}")
    set_seed(args.seed)
    tokens = torch.load(CACHE / "tokens.pt", weights_only=True, map_location="cpu")
    records = [tokens[r["id"]] for r in read_rows(RUN / "splits/content_groups" / (args.split+".jsonl"))]
    model, _ = load_qwen()
    wrapped = AlignedModel(model, args.arm)
    add_lora(model,seed=args.seed)
    if args.gradient_checkpointing:
        # Enable only when profiling shows it is needed: B300 memory should
        # otherwise avoid recomputation and hold useful larger microbatches.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = dict(vars(args), output=str(args.output), records=len(records), cache_identity=identity(),
        initialization_sha256=initialization_hashes(model),
        code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                     for name in ("runtime.py","fusion.py","kernels.py","train.py","cache.py")})
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != configuration:
        raise RuntimeError("Refusing incompatible output-directory reuse")
    config_path.write_text(json.dumps(configuration, indent=2)+"\n")
    if args.profile:
        probe_rows = read_rows(RUN/"splits/content_groups/train.jsonl")[:1]
        probe = collate(probe_rows,processor())
        standalone = load_native_vision()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            reference = wrapped.original_forward(probe["pixel_values"].cuda().bfloat16(),grid_thw=probe["image_grid_thw"].cuda()).last_hidden_state
            separate = standalone(probe["pixel_values"].cuda().bfloat16(),grid_thw=probe["image_grid_thw"].cuda()).last_hidden_state
        torch.testing.assert_close(reference,separate,atol=0,rtol=0)
        cached = feature("native",probe_rows[0]["image_sha256"])["native"].cuda()
        relative_error = ((reference.float()-cached.float()).norm()/reference.float().norm()).item()
        assert relative_error < .01,relative_error
        (args.output/"native_parity.json").write_text(json.dumps({"standalone_full_model_exact_parity":True,
            "cached_batch_vs_online_single_relative_l2":relative_error},indent=2)+"\n")
        del standalone,reference,separate,cached,probe
        wrapped.stage("B")
        longest = sorted(records, key=lambda r:r["input_ids"].numel(), reverse=True)
        measurements = []
        for size in (1,2,4,8,16,32):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch, context = make_batch(longest[:size], args.arm)
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                loss = wrapped.loss(batch, context)
                loss.backward()
                torch.cuda.synchronize()
                item = {"microbatch": size, "loss": loss.item(), "seconds": time.monotonic()-start,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                        "sequence_length": batch["input_ids"].shape[1]}
                measurements.append(item)
                print(json.dumps(item), flush=True)
                del loss
            except torch.OutOfMemoryError:
                measurements.append({"microbatch": size, "out_of_memory": True})
                break
        (args.output / "profile.json").write_text(json.dumps(measurements,indent=2)+"\n")
        return
    latest = args.output / "latest.pt"
    resume = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    completed = resume["epoch"] if resume else 0
    if completed == 4:
        print("All four epochs already complete", flush=True)
        return
    started = time.monotonic()
    global_step = resume["global_step"] if resume else 0
    for stage, epochs in (("A", [1]), ("B", [2,3,4])):
        if epochs[-1] <= completed:
            continue
        wrapped.stage(stage)
        opt = optimizer_for(model, stage)
        steps_per_epoch = math.ceil(len(records)/32)
        total_stage_steps = len(epochs)*steps_per_epoch
        warmup = max(1, math.ceil(total_stage_steps*.03))
        if resume is not None:
            # A->B needs the Stage-A key set, then enables the common LoRA set.
            wrapped.stage(resume["stage"])
            restore_trainable(model,resume["adapter"])
            wrapped.stage(stage)
            if resume["stage"] == stage:
                opt.load_state_dict(resume["optimizer"])
            torch.set_rng_state(resume["torch_rng"])
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
            random.setstate(resume["python_rng"])
            np.random.set_state(resume["numpy_rng"])
            resume = None
        stage_step = max(0,completed-epochs[0]+1)*steps_per_epoch
        for epoch in epochs:
            if epoch <= completed:
                continue
            order = epoch_order(records,args.seed,epoch)
            totals, observed = 0., 0
            torch.cuda.reset_peak_memory_stats()
            for begin in range(0,len(order),32):
                group = [records[i] for i in order[begin:begin+32]]
                bucket=math.ceil(max(r["input_ids"].numel() for r in group)/128)*128
                microbatch=args.short_microbatch if args.short_microbatch and bucket<=768 else args.microbatch
                supervised = sum((r["labels"]!=-100).sum().item() for r in group)
                opt.zero_grad(set_to_none=True)
                factor = (stage_step+1)/warmup if stage_step < warmup else .5*(1+math.cos(math.pi*(stage_step-warmup)/max(1,total_stage_steps-warmup)))
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"]*factor
                step_loss = 0.
                for j in range(0,len(group),microbatch):
                    batch, context = make_batch(group[j:j+microbatch],args.arm)
                    count = (batch["labels"]!=-100).sum().item()
                    loss = wrapped.loss(batch,context)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    (loss*(count/supervised)).backward()
                    step_loss += loss.item()*count/supervised
                    del loss,batch,context
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
                opt.step()
                global_step += 1
                stage_step += 1
                observed += len(group)
                totals += step_loss*len(group)
                event = {"epoch":epoch,"stage":stage,"global_step":global_step,"loss":step_loss,"grad_norm":norm.item(),
                         "microbatch":microbatch,"sequence_bucket":bucket,
                         "examples":observed,"elapsed_seconds":time.monotonic()-started,
                         "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
                with (args.output/"training.jsonl").open("a") as f:
                    f.write(json.dumps(event)+"\n")
                if global_step%10 == 0:
                    print(json.dumps(event),flush=True)
            assert observed == len(records)
            checkpoint = {"adapter":trainable_state(model),"optimizer":opt.state_dict(),"epoch":epoch,"stage":stage,
                          "global_step":global_step,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                          "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"configuration":configuration}
            atomic_save(checkpoint,args.output/f"epoch_{epoch}.pt")
            tmp_link = args.output/"latest.tmp"
            tmp_link.unlink(missing_ok=True)
            tmp_link.symlink_to(f"epoch_{epoch}.pt")
            os.replace(tmp_link,latest)
            print(json.dumps({"epoch_complete":epoch,"mean_loss":totals/observed}),flush=True)
            del checkpoint
        del opt
    (args.output/"complete.json").write_text(json.dumps({"epochs":4,"global_steps":global_step,"examples_per_epoch":len(records),
        "elapsed_seconds_this_process":time.monotonic()-started,"final_checkpoint":"epoch_4.pt"},indent=2)+"\n")


if __name__ == "__main__":
    main()
