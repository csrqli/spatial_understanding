"""Rebuild the audited MindCube-to-3DThinker reasoning join from public files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def rows(path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def normalized_path(path):
    return path.replace("data/other_all_image_resize/", "other_all_image/", 1)


def key(question, images, answer):
    return (re.sub(r"\s+", " ", question).strip(),
            tuple(normalized_path(x) for x in images), answer)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mindcube-train", type=Path, required=True)
    parser.add_argument("--thinker-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = {}
    for row in rows(args.mindcube_train):
        identity = key(row["question"], row["images"], row["gt_answer"])
        if identity in original:
            raise ValueError("Ambiguous raw MindCube identity")
        original[identity] = row
    if len(original) != 10000:
        raise ValueError(f"Expected 10,000 unique raw MindCube questions, got {len(original)}")
    accepted = []
    seen = set()
    for teacher in rows(args.thinker_jsonl):
        answer = teacher["answer"][:1]
        identity = key(teacher["text_input"], teacher["image_input"], answer)
        if identity not in original or identity in seen:
            raise ValueError("Missing or repeated exact question/image/answer match")
        seen.add(identity)
        trace = teacher["text_output"]
        match = re.search(r"<think>(.*?)</think>", trace, flags=re.S)
        if match is None or not match.group(1).strip():
            raise ValueError("Missing teacher rationale")
        answer_match = re.search(r"<answer>\s*([A-D])\.", trace, flags=re.S)
        if answer_match is None or answer_match.group(1) != answer:
            raise ValueError("Teacher's marked answer disagrees with MindCube")
        rationale = match.group(1).strip()
        accepted.append({
            "mind_id": "mind/" + original[identity]["id"],
            "teacher_idx": teacher["idx"],
            "answer": answer,
            "reasoning": rationale,
            "reasoning_chars": len(rationale),
            "source": "3DThinker-10K",
            "matched_by": "exact normalized question AND ordered source image filenames AND answer letter",
        })
    if len(accepted) != 10000 or len(seen) != len(original):
        raise ValueError("The public files did not yield a complete 10,000-question match")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in accepted:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "counts": dict(Counter(accepted=10000)),
        "raw_teacher_rows": 10000,
        "unique_accepted_questions": 10000,
        "source_sha256": {
            str(args.mindcube_train): sha256(args.mindcube_train),
            str(args.thinker_jsonl): sha256(args.thinker_jsonl),
        },
        "mean_reasoning_chars": sum(r["reasoning_chars"] for r in accepted) / len(accepted),
        "note": "Exact annotation identity checked; geometric validity of each rationale is not independently verified.",
    }
    args.output.with_name("reasoning_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
