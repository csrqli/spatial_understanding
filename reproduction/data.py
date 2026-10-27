"""Prepare OmniSpatial SFT and image-disjoint validation, without test labels."""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system_prompts import FORMAT_PROMPTS, SYS_PROMPTS


def question_text(info, prompt_type="none"):
    question = info["question"]
    if isinstance(question, list):
        question = " ".join(question)
    # These are text references to panels already composited into the PNG,
    # not additional processor image tokens. Keep their ordinal descriptions.
    question = question.replace("<image>", "(shown in the image)")
    text = SYS_PROMPTS[prompt_type] + "\n" + FORMAT_PROMPTS["re"] + "\n\n" + question
    for index, option in enumerate(info["options"]):
        if option:
            text += f"\n{chr(65 + index)}. {option.replace('<image>', '(shown in the image)')}"
    return text


def grouped_split(records, validation_fraction=0.1, seed=42):
    """Keep identical image bytes together, including differently named PNGs."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    groups = defaultdict(list)
    for record in records:
        groups[record["image_sha256"]].append(record)
    by_category = defaultdict(list)
    for digest, group in groups.items():
        category = min(record["sub_task_type"] for record in group)
        by_category[category].append(digest)
    rng = random.Random(seed)
    held_out = set()
    for category in sorted(by_category):
        digests = sorted(by_category[category])
        rng.shuffle(digests)
        count = min(len(digests) - 1, max(1, round(len(digests) * validation_fraction)))
        held_out.update(digests[:count])
    return (
        [r for r in records if r["image_sha256"] not in held_out],
        [r for r in records if r["image_sha256"] in held_out],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-type", choices=list(SYS_PROMPTS), default="none")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.output_dir}")
    source = args.train_root / "data.json"
    data = json.loads(source.read_text())
    records, seen, hashes = [], set(), {}
    for info in data:
        key = f"{info['task_type']}/{info['id']}"
        if key in seen:
            raise ValueError(f"Duplicate question: {key}")
        seen.add(key)
        answer = info["answer"]
        # Empty options can mean options embedded in the image, not missing data.
        if not isinstance(answer, int) or not 0 <= answer < 4:
            raise ValueError(f"Invalid answer: {key}")
        path = args.train_root / info["task_type"] / f"{info['id'].split('_')[0]}.png"
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "id": key, "image": str(path.resolve()), "image_sha256": hashes[path],
            "task_type": info["task_type"], "sub_task_type": info["sub_task_type"],
            "prompt": question_text(info, args.prompt_type), "answer": chr(65 + answer),
            "target": f"Answer: {chr(65 + answer)}",
        })
    train, validation = grouped_split(records, args.validation_fraction, args.seed)
    if not train or not validation:
        raise ValueError("Empty train or validation split")
    args.output_dir.mkdir(parents=True)
    for name, split in (("all", records), ("train", train), ("validation", validation)):
        with (args.output_dir / f"{name}.jsonl").open("x") as f:
            for record in split:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": args.seed, "validation_fraction": args.validation_fraction,
        "prompt_type": args.prompt_type, "split_unit": "sha256 of image bytes",
        "counts": {"all": len(records), "train": len(train), "validation": len(validation)},
        "categories": {name: dict(Counter(r["sub_task_type"] for r in split))
                       for name, split in (("all", records), ("train", train), ("validation", validation))},
        "test_data_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
