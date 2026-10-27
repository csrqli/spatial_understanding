"""Verify and summarize a complete O2-only OmniSpatial reasoning evaluation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .config import RUN


def read_jsonl(path):
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_and_merge(output):
    expected_rows = [row for row in read_jsonl(RUN / 'splits/test.jsonl')
                     if row['source'] == 'omni_test']
    expected = {row['id']: row for row in expected_rows}
    if len(expected_rows) != 1533 or len(expected) != 1533:
        raise RuntimeError('Official test must contain 1,533 unique IDs')
    aligned = output / 'O2'
    sources = ([aligned] if (aligned / 'metrics.json').exists() else
               [aligned / 'shard0', aligned / 'shard1'])
    observed = {}
    for folder in sources:
        metrics = json.loads((folder / 'metrics.json').read_text())
        rows = read_jsonl(folder / 'predictions.jsonl')
        if not metrics.get('coverage_verified') or metrics['n'] != len(rows):
            raise RuntimeError(f'Incomplete evaluation: {folder}')
        correct = sum(row['correct'] for row in rows)
        if abs(correct / len(rows) - metrics.get('micro_accuracy', metrics.get('accuracy', -1))) > 1e-12:
            raise RuntimeError(f'Accuracy mismatch: {folder}')
        for row in rows:
            rid = row['id']
            if rid not in expected or rid in observed:
                raise RuntimeError(f'Unknown or duplicate test ID: {rid}')
            reference = expected[rid]
            for key in ('answer', 'source', 'task_type', 'sub_task_type'):
                if row[key] != reference[key]:
                    raise RuntimeError(f'Test metadata mismatch: {rid} {key}')
            if row['correct'] != (row['predicted'] == row['answer']):
                raise RuntimeError(f'Wrong readout correctness flag: {rid}')
            if row['generated_correct'] != (row['generated_answer'] == row['answer']):
                raise RuntimeError(f'Wrong natural correctness flag: {rid}')
            observed[rid] = row
    if set(observed) != set(expected):
        raise RuntimeError(f'Missing {len(expected) - len(observed)} official test IDs')
    ordered = [observed[row['id']] for row in expected_rows]
    tasks = defaultdict(list)
    for row in ordered:
        tasks[row['task_type']].append(row)
    correct = sum(row['correct'] for row in ordered)
    natural = sum(row['generated_correct'] for row in ordered)
    summary = {
        'n': 1533, 'correct': correct, 'accuracy': correct / 1533,
        'natural_generation_correct': natural,
        'natural_generation_accuracy': natural / 1533,
        'natural_invalid_answers': sum(row['generated_answer'] is None for row in ordered),
        'truncated': sum(row['truncated'] for row in ordered),
        'coverage_verified': True, 'training_seed': 42,
        'scoring': '8192-token reasoning followed by fixed valid-letter readout; natural generation separate',
        'tasks': {task: {'n': len(rows), 'correct': sum(row['correct'] for row in rows),
                         'accuracy': sum(row['correct'] for row in rows) / len(rows)}
                  for task, rows in sorted(tasks.items())},
    }
    if len(sources) == 2:
        aligned.mkdir(parents=True, exist_ok=True)
        with (aligned / 'predictions.jsonl').open('w') as destination:
            for row in ordered:
                destination.write(json.dumps(row) + '\n')
        (aligned / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    (aligned / 'verified_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Verified O2-only OmniSpatial evaluation', '',
             f'Valid-letter readout: **{correct}/1,533 = {correct/1533:.2%}**.',
             f'Natural generated answer: **{natural}/1,533 = {natural/1533:.2%}**.',
             f'Invalid natural answers: {summary["natural_invalid_answers"]}; capped generations: {summary["truncated"]}.',
             '', '| Category | Correct / N | Accuracy |', '|---|---:|---:|']
    for task, values in summary['tasks'].items():
        lines.append(f'| {task} | {values["correct"]}/{values["n"]} | {values["accuracy"]:.2%} |')
    lines += ['', 'All 1,533 official test IDs were checked exactly once. This public test was inspected during development, so this is a follow-up result.', '']
    (aligned / 'VERIFIED_REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_and_merge(args.output), indent=2))


if __name__ == '__main__':
    main()
