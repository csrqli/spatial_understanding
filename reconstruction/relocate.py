"""Rewrite this project's original absolute paths in a copied source tree.

Run only on the destination copy, before generating any caches or checkpoints.
Existing run receipts contain source hashes and must never be relocated in place.
"""
import argparse
from pathlib import Path


OLD_ROOT = "/dataset1/yu025/code_share/OmniSpatial"
OLD_DATA = "/dataset1/yu025/datasets/spatial_alignment_v2"
OLD_MIND = "/dataset1/yu025/datasets/3dthinker/MindCube/extracted/data"
OLD_OMNI = "/dataset1/yu025/datasets/omnispatial/OmniSpatial/extracted/test/OmniSpatial-test"
OLD_QWEN = "/dataset1/yu025/models/Qwen3.5-9B"
OLD_OMEGA = "/dataset1/yu025/models/vlm3r_omega/omega/vggt_omega_1b_416_reproduce.pt"
OLD_VGGT = "/dataset1/yu025/models/3dthinker/VGGT-1B/model.safetensors"
OLD_ENV = "/dataset1/yu025/code_share/envs/qwen35-omega"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--env-prefix", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    if not (project / "research/qwen35_omega_v2/config.py").exists():
        raise FileNotFoundError("Pass the copied project directory with research/ inside")
    mapping = {
        OLD_MIND: str(project / "datasets/3dthinker/MindCube/extracted/data"),
        OLD_OMNI: str(project / "datasets/omnispatial/extracted/test/OmniSpatial-test"),
        OLD_DATA: str(project / "datasets/spatial_alignment_v2"),
        OLD_QWEN: str(project / "models/Qwen3.5-9B"),
        OLD_OMEGA: str(project / "models/vggt-omega/vggt_omega_1b_416_reproduce.pt"),
        OLD_VGGT: str(project / "models/VGGT-1B/model.safetensors"),
        OLD_ENV: str(args.env_prefix.resolve()),
        OLD_ROOT: str(project),
    }
    selected = list((project / "research/qwen35_omega").glob("*.py"))
    selected += list((project / "research/qwen35_omega_v2").glob("*.py"))
    selected += list((project / "research/qwen35_omega").glob("*.sbatch"))
    selected += list((project / "research/qwen35_omega_v2").glob("*.sbatch"))
    selected += [project / "reproduction/slurm_b300_env.sh"]
    changed = []
    for path in selected:
        if not path.exists():
            continue
        before = path.read_text()
        after = before
        for old, new in mapping.items():
            after = after.replace(old, new)
        if after != before:
            path.write_text(after)
            changed.append(str(path.relative_to(project)))
    for name in ("research/qwen35_omega_v2/config.py", "research/qwen35_omega/runtime.py"):
        if OLD_ROOT in (project / name).read_text() or OLD_QWEN in (project / name).read_text():
            raise RuntimeError(f"Unrelocated source path in {name}")
    print("Relocated", len(changed), "source files:")
    print("\n".join(changed))


if __name__ == "__main__":
    main()
