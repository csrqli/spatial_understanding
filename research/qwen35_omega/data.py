"""Audit OmniSpatial media and freeze content-grouped training-side partitions.

Only training annotations are read. The test root is inspected for PNG contents
to exclude exact overlap; test labels are never loaded by this program.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproduction.data import question_text


def canonical_rgb(image):
    if image.mode == "RGBA" or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return image.convert("RGB")


def letterbox_square(image):
    """Preserve every source pixel before an encoder's documented resize.

    Returns the padded RGB image, a pixel-validity mask, and the exact crop-free
    transform. This is an explicit OmniSpatial protocol change, not upstream
    Omega's default extreme-aspect crop.
    """
    image = canonical_rgb(image)
    width, height = image.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    padded = Image.new("RGB", (side, side), "white")
    padded.paste(image, (left, top))
    mask = Image.new("L", (side, side), 0)
    mask.paste(255, (left, top, left + width, top + height))
    return padded, mask, {"source_size_wh": [width, height], "padded_size_wh": [side, side],
                          "content_box_xyxy": [left, top, left + width, top + height],
                          "fill_rgb": [255, 255, 255], "crop_applied": False}


def inspect_image(path):
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as original:
        original.verify()
    with Image.open(io.BytesIO(data)) as original:
        image = canonical_rgb(original)
        image.load()
        width, height = image.size
        digest = hashlib.sha256(f"RGB:{width}:{height}:".encode() + image.tobytes()).hexdigest()
        return {"path": str(path.resolve()), "file_sha256": hashlib.sha256(data).hexdigest(),
                "image_sha256": digest, "width": width, "height": height,
                "aspect_ratio": width / height, "source_mode": original.mode,
                "exif_orientation": original.getexif().get(274, 1),
                "omega_extreme_aspect_crop": width / height > 2 or height / width > 2}


def split_groups(records, seed=20261024, development_fraction=0.15, confirmation_fraction=0.10):
    groups = defaultdict(list)
    for record in records:
        groups[record["image_sha256"]].append(record)
    strata = defaultdict(list)
    for digest, rows in groups.items():
        strata[min((r["task_type"], r["sub_task_type"]) for r in rows)].append(digest)
    rng, assignment = random.Random(seed), {}
    for category in sorted(strata):
        digests = sorted(strata[category])
        rng.shuffle(digests)
        n = len(digests)
        dev = max(1, round(n * development_fraction))
        conf = max(1, round(n * confirmation_fraction))
        if n - dev - conf < 1:
            raise ValueError(f"Insufficient image groups for three partitions: {category}, {n}")
        for index, digest in enumerate(digests):
            assignment[digest] = "development" if index < dev else "confirmation" if index < dev + conf else "train"
    return {name: [r for r in records if assignment[r["image_sha256"]] == name]
            for name in ("train", "development", "confirmation")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--test-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    source = args.train_root / "data.json"
    annotations = json.loads(source.read_text())
    source_images = sorted({args.train_root / row["task_type"] / (row["id"].split("_")[0] + ".png") for row in annotations})
    test_images = sorted(args.test_image_root.glob("*/*.png"))
    if not annotations or not test_images:
        raise ValueError("Missing training annotations or test images for exclusion audit")
    print(f"Auditing {len(source_images)} training images and {len(test_images)} test images; test labels are not read.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        train_media = list(executor.map(inspect_image, source_images))
        test_media = list(executor.map(inspect_image, test_images))
    by_path = {r["path"]: r for r in train_media}
    test_digests = {r["image_sha256"] for r in test_media}
    seen_ids, records, excluded = set(), [], []
    targets, signatures = defaultdict(set), Counter()
    for info in annotations:
        key = f"{info['task_type']}/{info['id']}"
        if key in seen_ids:
            raise ValueError(f"Duplicate QA ID: {key}")
        seen_ids.add(key)
        if not isinstance(info["answer"], int) or not 0 <= info["answer"] < 4:
            raise ValueError(f"Invalid answer: {key}")
        path = (args.train_root / info["task_type"] / (info["id"].split("_")[0] + ".png")).resolve()
        media = by_path[str(path)]
        record = {"id": key, "image": str(path), "image_sha256": media["image_sha256"],
                  "image_file_sha256": media["file_sha256"], "task_type": info["task_type"],
                  "sub_task_type": info["sub_task_type"], "prompt": question_text(info, "none"),
                  "answer": chr(65 + info["answer"]), "target": f"Answer: {chr(65 + info['answer'])}"}
        signature = (record["image_sha256"], record["prompt"])
        targets[signature].add(record["answer"])
        signatures[signature] += 1
        if record["image_sha256"] in test_digests:
            excluded.append(dict(record, exclusion_reason="exact_decoded_pixel_overlap_with_test_image"))
        else:
            records.append(record)
    conflicts = [key for key, values in targets.items() if len(values) > 1]
    if conflicts:
        raise ValueError(f"Conflicting answer targets for {len(conflicts)} identical image/prompt pairs")
    splits = split_groups(records, args.seed)
    image_sets = {name: {r["image_sha256"] for r in rows} for name, rows in splits.items()}
    assert not image_sets["train"] & image_sets["development"]
    assert not image_sets["train"] & image_sets["confirmation"]
    assert not image_sets["development"] & image_sets["confirmation"]
    assert len(records) + len(excluded) == len(annotations)
    all_tasks = {r["task_type"] for r in records}
    for name, rows in splits.items():
        if {r["task_type"] for r in rows} != all_tasks:
            raise ValueError(f"Missing main task category in {name}")
    args.output.mkdir(parents=True)
    outputs = dict(splits, full_train=records, excluded_test_overlap=excluded)
    hashes = {}
    for name, rows in outputs.items():
        p = args.output / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    media_path = args.output / "media_audit.json"
    media_path.write_text(json.dumps({"training": train_media, "test_images_for_overlap_only": test_media}, indent=2) + "\n")
    hashes[media_path.name] = hashlib.sha256(media_path.read_bytes()).hexdigest()
    manifest = {"schema_version": "qwen35_omega_omnispatial_data_v1", "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "split_seed": args.seed,
                "split_unit": "SHA-256 of canonical decoded RGB pixels and dimensions; white alpha compositing",
                "test_labels_read": False, "test_image_contents_used_only_for_overlap_exclusion": True,
                "requested_development_group_fraction": 0.15, "requested_confirmation_group_fraction": 0.10,
                "source_qa_count": len(annotations), "source_image_count": len(source_images),
                "test_image_count": len(test_images), "test_overlap_qa_excluded": len(excluded),
                "duplicate_exact_image_prompt_excess": sum(n - 1 for n in signatures.values()),
                "counts": {name: len(rows) for name, rows in outputs.items()},
                "image_groups": {name: len({r["image_sha256"] for r in rows}) for name, rows in outputs.items()},
                "main_categories": {name: dict(Counter(r["task_type"] for r in rows)) for name, rows in outputs.items()},
                "sub_categories": {name: dict(Counter(r["sub_task_type"] for r in rows)) for name, rows in outputs.items()},
                "training_images_cropped_by_official_omega_extreme_aspect_rule": sum(r["omega_extreme_aspect_crop"] for r in train_media),
                "source_scene_or_template_independence_established": False,
                "limitations": ["Exact image contents are grouped; independent physical scenes or templates are not asserted.",
                                "Near-duplicate scenes, foundation-model pretraining overlap and benchmark contamination are not resolved.",
                                "Sparse subcategory support must be reported; training records are not duplicated to balance categories."],
                "artifact_sha256": hashes}
    p = args.output / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    p.with_suffix(".sha256").write_text(hashlib.sha256(p.read_bytes()).hexdigest() + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "artifact_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
