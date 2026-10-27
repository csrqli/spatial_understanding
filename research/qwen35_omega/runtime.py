"""Qwen3.5 integration with explicit frozen feature and adapter contracts."""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .data import canonical_rgb, letterbox_square
from .fusion import SpatialAlignment, masked_grid_pool

ROOT = Path(os.environ.get("Q35_REPO_ROOT",Path(__file__).resolve().parents[2]))
RUN = ROOT / "results/qwen35_omega_omnispatial/20261024_revision"
BASE = "/dataset1/yu025/models/Qwen3.5-9B"
OMEGA = "/dataset1/yu025/models/vlm3r_omega/omega/vggt_omega_1b_416_reproduce.pt"
VGGT = "/dataset1/yu025/models/3dthinker/VGGT-1B/model.safetensors"
ARMS = {"B1": None, "R0": None, "G1": "vggt", "O1": "omega", "G2": "vggt", "OP": "omega", "O2": "omega"}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def processor():
    p = AutoProcessor.from_pretrained(BASE, local_files_only=True)
    p.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1003520}
    p.tokenizer.padding_side = "left"
    return p


def collate(rows, p, training=True):
    texts, images, suffixes = [], [], []
    for r in rows:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": r["prompt"]}]}]
        prompt = p.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if training:
            text = p.apply_chat_template(messages + [{"role": "assistant", "content": r["target"]}],
                                         tokenize=False, add_generation_prompt=False, enable_thinking=False)
            if not text.startswith(prompt):
                raise ValueError("Assistant template does not extend prompt")
            suffixes.append(p.tokenizer(text[len(prompt):], add_special_tokens=False).input_ids)
        else:
            text = prompt
        texts.append(text)
        with Image.open(r["image"]) as im:
            images.append(canonical_rgb(im))
    batch = dict(p(text=texts, images=images, padding=True, return_tensors="pt"))
    if batch["input_ids"].shape[1] > 4096:
        raise ValueError("Sequence exceeds 4096; no silent truncation")
    if training:
        labels = torch.full_like(batch["input_ids"], -100)
        for i, suffix in enumerate(suffixes):
            if not suffix or batch["input_ids"][i, -len(suffix):].tolist() != suffix:
                raise ValueError("Assistant suffix token mismatch")
            labels[i, -len(suffix):] = torch.tensor(suffix)
        batch["labels"] = labels
    return batch


def load_qwen():
    from .kernels import configure
    configure()
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": f"cuda:{torch.cuda.current_device()}"}, local_files_only=True,
        attn_implementation="sdpa", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Base load failure: {info}")
    allow = set(json.loads((RUN / "manifests/native_interface_check.json").read_text())["unexpected_keys"])
    if set(info.get("unexpected_keys", [])) - allow:
        raise RuntimeError(f"Unallowlisted base tensors: {info}")
    model.requires_grad_(False).eval()
    bound_patch_convolution(model.model.visual)
    model.config.use_cache = False
    info = json.loads(json.dumps(info,default=lambda value: sorted(value) if isinstance(value,set) else str(value)))
    return model, info


def load_native_vision():
    """Load exactly the native vision keys, without materializing the 9B LM."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    config = AutoConfig.from_pretrained(BASE,local_files_only=True).vision_config
    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=False):
        visual = Qwen3_5VisionModel(config)
    index = json.loads((Path(BASE)/"model.safetensors.index.json").read_text())["weight_map"]
    selected = {k:v for k,v in index.items() if k.startswith("model.visual.")}
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(str(Path(BASE)/shard),framework="pt",device="cpu") as f:
            for key in selected:
                if selected[key]==shard:
                    state[key.removeprefix("model.visual.")] = f.get_tensor(key)
    visual.load_state_dict(state,strict=True,assign=True)
    # Checkpoint parameters already carry BF16. Preserve FP32 nonpersistent
    # rotary buffers, as from_pretrained does, instead of casting the module.
    visual = visual.requires_grad_(False).eval().cuda()
    bound_patch_convolution(visual)
    return visual


def bound_patch_convolution(visual):
    """Equivalent patch GEMM, preserving Conv3d's separate BF16 bias addition."""
    embed = visual.patch_embed
    original = embed.forward
    def forward(module, hidden_states):
        weight=module.proj.weight
        patches=hidden_states.reshape(-1,weight[0].numel()).to(weight.dtype)
        # torch Conv3d adds bias after the cuDNN convolution output has rounded
        # to BF16. A fused linear bias would round only once and is NOT equal.
        projected=F.linear(patches,weight.flatten(1),None)
        return projected+module.proj.bias.to(projected.dtype)
    embed.original_conv_forward = original
    embed.forward = types.MethodType(forward,embed)


def load_spatial(kind):
    """Load the entire source checkpoint strictly, then retain only its aggregator."""
    vendor = ROOT / "research/vlm3r_omega/vendor"
    if kind == "omega":
        sys.path.insert(0, str(vendor / "vggt-omega"))
        from vggt_omega.models.vggt_omega import VGGTOmega
        model = VGGTOmega()
        state = torch.load(OMEGA, map_location="cpu", weights_only=True, mmap=True)
    else:
        sys.path.insert(0, str(vendor / "vggt"))
        from vggt.models.vggt import VGGT as VGGTModel
        from safetensors.torch import load_file
        model = VGGTModel()
        state = load_file(VGGT)
    model.load_state_dict(state, strict=True)
    aggregator = model.aggregator
    # Match upstream inference: keep checkpoint weights (including normalization)
    # in FP32, autocast matrix operations during the forward pass.
    aggregator.requires_grad_(False).eval().to("cuda")
    return aggregator


def prepare_spatial(row, kind):
    side, patch = (416, 16) if kind == "omega" else (518, 14)
    with Image.open(row["image"]) as im:
        padded, mask, _ = letterbox_square(im)
    image = torch.from_numpy(np.array(padded.resize((side, side), Image.Resampling.BICUBIC))).permute(2, 0, 1).float() / 255
    mask = torch.from_numpy(np.array(mask.resize((side, side), Image.Resampling.BOX))).float() / 255
    return image, F.avg_pool2d(mask[None, None], patch, patch)[0, 0]


@torch.no_grad()
def spatial_features(encoder, kind, rows, prepared=None):
    side, patch = (416, 16) if kind == "omega" else (518, 14)
    if prepared is None:
        values = [prepare_spatial(row,kind) for row in rows]
        prepared = (torch.stack([v[0] for v in values]),torch.stack([v[1] for v in values]))
    images = prepared[0].cuda(non_blocking=True)[:, None]
    weights = prepared[1].cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, start = encoder(images)
    final = tokens[-1][:, 0]
    expected = 17 if kind == "omega" else 5
    if start != expected or final.shape[-1] != 2048:
        raise RuntimeError("Unexpected auxiliary token layout")
    grid = final[:, start:].reshape(len(rows), side // patch, side // patch, 2048)
    local, local_valid = masked_grid_pool(grid.float(), weights, 14)
    pooled, pooled_valid = masked_grid_pool(grid.float(), weights, 4)
    camera = final[:, :1]
    result = {"local": torch.cat((local, camera), 1).bfloat16(),
              "local_valid": torch.cat((local_valid, torch.ones(len(rows), 1, device="cuda", dtype=torch.bool)), 1),
              "pooled_scene": torch.cat((camera, pooled), 1).bfloat16(),
              "pooled_valid": torch.cat((torch.ones(len(rows), 1, device="cuda", dtype=torch.bool), pooled_valid), 1)}
    if kind == "omega":
        result["register_scene"] = final[:, :17].bfloat16().contiguous()
    return result


class AlignedModel:
    """Explicit per-call context; no cross-example feature mixing or stale generation data."""
    def __init__(self, model, arm):
        self.model, self.arm, self.context = model, arm, None
        self.visual = model.model.visual
        self.original_forward = self.visual.forward
        self.alignment = None if ARMS[arm] is None else SpatialAlignment(scene_branch=arm in ("G2", "OP", "O2")).cuda()
        if self.alignment is not None:
            model.add_module("spatial_alignment", self.alignment)
        owner = self

        def forward(visual, hidden_states, grid_thw, **kwargs):
            if owner.context is None:
                raise RuntimeError("Visual features requested without explicit image context")
            context = owner.context
            if "native" in context:
                native = context["native"]
                if not torch.equal(context["grid"].to(grid_thw), grid_thw):
                    raise RuntimeError("Cache native grid mismatch")
            else:
                # Frozen vision only: the learned merger is invoked below with autograd.
                with torch.no_grad():
                    native = owner.original_forward(hidden_states, grid_thw=grid_thw, **kwargs).last_hidden_state
            sizes = grid_thw.prod(-1).tolist()
            pieces = native.split(sizes)
            if owner.alignment is not None:
                # Pad native tokens for efficient attention, then restore EXACT original order.
                padded = nn.utils.rnn.pad_sequence(pieces, batch_first=True)
                valid = torch.arange(padded.shape[1], device=padded.device)[None] < torch.tensor(sizes, device=padded.device)[:, None]
                scene_key = "register_scene" if owner.arm == "O2" else "pooled_scene"
                scene = context.get(scene_key)
                scene_valid = torch.ones(scene.shape[:2], dtype=torch.bool, device=scene.device) if owner.arm == "O2" else context.get("pooled_valid")
                fused = owner.alignment(padded, context["local"], scene=scene, visual_valid=valid,
                                        local_valid=context["local_valid"], scene_valid=scene_valid)
                native = torch.cat([fused[i, :n] for i, n in enumerate(sizes)])
            merged = visual.merger(native)
            return BaseModelOutputWithPooling(last_hidden_state=native, pooler_output=merged)
        self.visual.forward = types.MethodType(forward, self.visual)

    def stage(self, stage):
        self.model.requires_grad_(False)
        self.visual.merger.requires_grad_(True).float()
        if self.alignment is not None:
            self.alignment.requires_grad_(True).float()
        if stage == "B":
            for name, p in self.model.named_parameters():
                if "lora_" in name:
                    p.requires_grad_(True)
                    p.data = p.data.float()
        self.model.train()
        self.visual.eval()
        if self.alignment is not None:
            self.alignment.train()
        self.visual.merger.train()

    def loss(self, batch, context):
        self.context = context
        batch = {k: v.cuda() for k, v in batch.items()}
        labels = batch.pop("labels")
        # Compute only logits preceding supervised tokens; CE exactly equals the
        # full-logit assistant-only loss, without a batch×sequence×vocab allocation.
        shifted = F.pad(labels, (0, 1), value=-100)[:, 1:]
        indices = torch.where((shifted != -100).any(0))[0]
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(**batch, logits_to_keep=indices, use_cache=False)
                loss = F.cross_entropy(out.logits.float().reshape(-1, out.logits.shape[-1]),
                                       shifted[:, indices].reshape(-1))
            return loss
        finally:
            self.context = None

    @torch.no_grad()
    def generate(self, batch, context, max_new_tokens=512):
        self.context = context
        batch = {k: v.cuda() for k, v in batch.items() if k != "labels"}
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        finally:
            self.context = None


def add_lora(model, seed=42):
    from peft import LoraConfig, inject_adapter_in_model
    import peft.tuners.lora.torchao as peft_torchao
    names = json.loads((RUN / "manifests/native_interface_check.json").read_text())["proposed_stage_b_lora_module_names"]
    modules = dict(model.named_modules())
    if any(name not in modules for name in names):
        raise RuntimeError("LoRA module contract changed")
    if any(type(modules[name]) is not nn.Linear for name in names):
        raise RuntimeError("This study requires plain, unquantized Linear projections")
    # The inherited env contains torchao 0.11, which PEFT 0.21 rejects even
    # for ordinary Linear modules. This run never uses torchao/quantization;
    # skip that optional dispatch without altering the shared installation.
    peft_torchao.is_torchao_available = lambda: False
    # Keep the common language adapter initialization independent of how many
    # random parameters each arm's spatial fusion allocated beforehand.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed+100000)
        inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=names, bias="none"), model)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(state):
        raise RuntimeError(f"Adapter key mismatch: missing={expected-set(state)}, unexpected={set(state)-expected}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                if p.shape != state[n].shape:
                    raise RuntimeError(f"Adapter shape mismatch: {n}")
                p.copy_(state[n])
