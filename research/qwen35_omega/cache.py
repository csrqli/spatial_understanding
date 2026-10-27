"""Immutable frozen-feature caches and checked assistant-token records."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gc
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from PIL import Image
from .data import canonical_rgb
from .runtime import BASE, OMEGA, VGGT, ROOT, RUN, collate, load_native_vision, load_spatial, processor, read_rows, spatial_features, prepare_spatial

CACHE = RUN / "cache_v1"


class SpatialImages(torch.utils.data.Dataset):
    def __init__(self,rows,kind):
        self.rows,self.kind = rows,kind

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        return prepare_spatial(self.rows[index],self.kind)


class NativeCollator:
    def __init__(self,p):
        self.image_processor=p.image_processor

    def __call__(self,rows):
        images=[]
        for row in rows:
            with Image.open(row["image"]) as image:
                images.append(canonical_rgb(image))
        return dict(self.image_processor(images=images,return_tensors="pt"))


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def identity():
    return {"protocol": "single-image-letterbox-v1", "qwen_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "omega_sha256": "e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896",
            "vggt_sha256": "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e",
            "qwen_size": [65536, 1003520], "omega_size": 416, "vggt_size": 518,
            "native_layer": "last_hidden_state_before_native_merger", "spatial_layer": "final_concat_frame_global",
            "dtype": "bfloat16", "T": 1,
            "data_manifest_sha256": hashlib.sha256((RUN / "splits/content_groups/manifest.json").read_bytes()).hexdigest()}


def initialize():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "identity.json"
    payload = json.dumps(identity(), sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x") as out:
            out.write(payload)
    except FileExistsError:
        if path.read_text() != payload:
            raise RuntimeError("Incompatible cache identity")


def tokenize_record(row, p):
    train = collate([row], p, training=True)
    generation = collate([row], p, training=False)
    length = generation["input_ids"].shape[1]
    if not torch.equal(train["input_ids"][0, :length], generation["input_ids"][0]):
        raise ValueError("Train/generation prefix mismatch")
    return {"id": row["id"], "image_sha256": row["image_sha256"], "grid": train["image_grid_thw"],
            "input_ids": train["input_ids"][0], "labels": train["labels"][0],
            "mm_token_type_ids": train["mm_token_type_ids"][0], "prompt_length": length}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("tokens", "native", "omega", "vggt"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90,repeat=True)
    initialize()
    rows = read_rows(RUN / "splits/content_groups/full_train.jsonl")
    start = time.monotonic()
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    p = processor() if args.kind in ("tokens", "native") else None
    if args.kind == "tokens":
        path = CACHE / "tokens.pt"
        if path.exists():
            raise FileExistsError(path)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            values = []
            for i, value in enumerate(pool.map(lambda r: tokenize_record(r, p), rows)):
                values.append(value)
                if (i+1) % 200 == 0:
                    print(f"tokens {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}", flush=True)
        atomic_save({r["id"]: r for r in values}, path)
        report = {"count": len(values), "max_sequence_length": max(v["input_ids"].numel() for v in values),
                  "min_supervised_tokens": min((v["labels"]!=-100).sum().item() for v in values)}
    else:
        unique = {r["image_sha256"]: r for r in rows}
        pending = [r for key,r in sorted(unique.items()) if not (CACHE / args.kind / f"{key}.pt").exists()]
        if args.kind == "native":
            encoder = load_native_vision()
        else:
            encoder = load_spatial(args.kind)
        encoder.eval()
        print(f"{args.kind} model loaded; pending={len(pending)}",flush=True)
        torch.cuda.reset_peak_memory_stats()
        prepared_batches = iter(torch.utils.data.DataLoader(SpatialImages(pending,args.kind),batch_size=args.batch_size,
            num_workers=args.workers,pin_memory=True,prefetch_factor=2)) if args.kind != "native" else None
        native_batches = iter(torch.utils.data.DataLoader(pending,batch_size=args.batch_size,
            collate_fn=NativeCollator(p),num_workers=args.workers,pin_memory=True,prefetch_factor=2)) if args.kind == "native" else None
        for i in range(0, len(pending), args.batch_size):
            batch_rows = pending[i:i+args.batch_size]
            if args.kind == "native":
                batch = next(native_batches)
                grid = batch["image_grid_thw"].cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    native = encoder(batch["pixel_values"].cuda().bfloat16(), grid_thw=grid).last_hidden_state
                pieces = native.split(grid.prod(-1).tolist())
                values = [{"native": v.cpu().contiguous(), "grid": g.cpu()[None]} for v,g in zip(pieces,grid)]
            else:
                features = spatial_features(encoder, args.kind, batch_rows,prepared=next(prepared_batches))
                values = [{k: v[j].cpu().contiguous() for k,v in features.items()} for j in range(len(batch_rows))]
            for row,value in zip(batch_rows, values):
                value["image_sha256"] = row["image_sha256"]
                value["cache_identity"] = identity()
                atomic_save(value, CACHE / args.kind / (row["image_sha256"] + ".pt"))
            if i == 0:
                faulthandler.cancel_dump_traceback_later()
            if i % (args.batch_size*10) == 0:
                print(f"{args.kind} {i+len(batch_rows)}/{len(pending)} elapsed={time.monotonic()-start:.1f} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        assert all((CACHE / args.kind / f"{key}.pt").exists() for key in unique)
        report = {"image_groups": len(unique), "newly_extracted": len(pending), "batch_size": args.batch_size,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
    report.update(kind=args.kind, elapsed_seconds=time.monotonic()-start, code_sha256=code_hashes, identity=identity())
    (CACHE / (args.kind+"_complete.json")).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
