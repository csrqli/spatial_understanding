"""Deterministic answer scoring; test access gated on all declared full refits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
import torch
from .cache import CACHE
from .data import inspect_image
from .runtime import (ARMS, RUN, AlignedModel, add_lora, collate, load_qwen, load_spatial,
                      processor, read_rows, restore_trainable, spatial_features)
from .train import make_batch


def parse_answer(text):
    explicit = re.findall(r"\bAnswer\s*:\s*([A-D])\b",text,flags=re.I)
    if len(explicit) == 1:
        return explicit[0].upper()
    if not explicit and re.fullmatch(r"\s*([A-D])[.\s]*", text, flags=re.I):
        return text.strip()[0].upper()
    return None


def metrics(predictions):
    by_task, by_sub = defaultdict(list), defaultdict(list)
    for p in predictions:
        by_task[p["task_type"]].append(p["correct"])
        by_sub[p["sub_task_type"]].append(p["correct"])
    def summarize(groups):
        return {k:{"n":len(v),"accuracy":sum(v)/len(v)} for k,v in sorted(groups.items())}
    tasks,subtasks = summarize(by_task),summarize(by_sub)
    return {"n":len(predictions),"macro_accuracy":sum(v["accuracy"] for v in tasks.values())/len(tasks),
            "micro_accuracy":sum(p["correct"] for p in predictions)/len(predictions),
            "invalid":sum(p["parsed"] is None for p in predictions),"tasks":tasks,"subtasks":subtasks}


def test_rows():
    expected = [RUN/"full"/f"{arm}_seed{seed}"/"complete.json" for arm in ("B1","G2","OP","O2") for seed in (42,43,44)]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"Official test locked until all 12 full refits complete: {missing}")
    from reproduction.data import question_text
    root = Path("/dataset1/yu025/datasets/omnispatial/OmniSpatial/extracted/test/OmniSpatial-test")
    rows = []
    for r in json.loads((root/"data.json").read_text()):
        image = root/r["task_type"]/(r["id"].split("_")[0]+".png")
        rows.append({"id":f"{r['task_type']}/{r['id']}","image":str(image),"image_sha256":inspect_image(image)["image_sha256"],
                     "prompt":question_text(r,"none"),"answer":chr(65+r["answer"]),
                     "task_type":r["task_type"],"sub_task_type":r["sub_task_type"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",choices=list(ARMS),required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--split",choices=("development","confirmation","test"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=8)
    args = parser.parse_args()
    if (args.arm == "R0") != (args.checkpoint is None):
        raise ValueError("Only R0 omits a trained checkpoint")
    if (args.output/"metrics.json").exists():
        raise FileExistsError("Evaluation already complete")
    rows = test_rows() if args.split == "test" else read_rows(RUN/"splits/content_groups"/(args.split+".jsonl"))
    p = processor()
    model,_ = load_qwen()
    wrapped = AlignedModel(model,args.arm)
    if args.checkpoint:
        saved = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if saved["configuration"]["arm"] != args.arm:
            raise ValueError("Checkpoint arm mismatch")
        add_lora(model,seed=saved["configuration"]["seed"])
        wrapped.stage(saved["stage"])
        restore_trainable(model,saved["adapter"])
        del saved
    model.eval()
    encoder = load_spatial(ARMS[args.arm]) if args.split == "test" and ARMS[args.arm] else None
    tokens = torch.load(CACHE/"tokens.pt",weights_only=True,map_location="cpu") if args.split != "test" else None
    args.output.mkdir(parents=True,exist_ok=True)
    config = {"arm":args.arm,"split":args.split,"checkpoint":str(args.checkpoint),"batch_size":args.batch_size,
              "checkpoint_size":args.checkpoint.stat().st_size if args.checkpoint else None,
              "checkpoint_mtime_ns":args.checkpoint.stat().st_mtime_ns if args.checkpoint else None,
              "max_new_tokens":512,"thinking":False,"do_sample":False}
    config_path = args.output/"configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError("Cannot resume evaluation with changed configuration")
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    predictions = read_rows(args.output/"predictions.jsonl") if (args.output/"predictions.jsonl").exists() else []
    if [r["id"] for r in predictions] != [r["id"] for r in rows[:len(predictions)]]:
        raise RuntimeError("Partial evaluation IDs do not match the requested split")
    start = time.monotonic()
    for begin in range(len(predictions),len(rows),args.batch_size):
        group = rows[begin:begin+args.batch_size]
        if tokens is not None:
            batch,context = make_batch([tokens[r["id"]] for r in group],args.arm,training=False)
        else:
            batch = collate(group,p,training=False)
            context = spatial_features(encoder,ARMS[args.arm],group) if encoder is not None else {}
        generated = wrapped.generate(batch,context)
        texts = p.batch_decode(generated[:,batch["input_ids"].shape[1]:],skip_special_tokens=True)
        new_ids = generated[:,batch["input_ids"].shape[1]:].tolist()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos,list) else [eos])
        for row,text,token_ids in zip(group,texts,new_ids):
            parsed = parse_answer(text)
            result = {k:row[k] for k in ("id","image_sha256","task_type","sub_task_type","answer")}
            result.update(text=text,parsed=parsed,correct=parsed==row["answer"],generated_token_ids=token_ids,
                          hit_token_limit=len(token_ids)>=512 and not any(t in eos for t in token_ids))
            predictions.append(result)
            with (args.output/"predictions.jsonl").open("a") as f:
                f.write(json.dumps(result,ensure_ascii=False)+"\n")
        if begin%(args.batch_size*10)==0:
            print(f"evaluated {len(predictions)}/{len(rows)}",flush=True)
    report = metrics(predictions)
    report.update(arm=args.arm,split=args.split,checkpoint=str(args.checkpoint),elapsed_seconds=time.monotonic()-start,
                  decoding={"do_sample":False,"thinking":False,"max_new_tokens":512})
    (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
