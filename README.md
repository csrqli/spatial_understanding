# OmniSpatial O2 and V3 reproduction source

This folder is the small, source-only handoff for the completed seed-42 O2 method and its planned V3 rationale continuation. Start with [HANDOFF.md](HANDOFF.md) for downloads, environment setup, relocation, Slurm commands, and full-test evaluation. The earlier research goal and plan are in [docs/RESEARCH_GOAL_AND_PLAN.md](docs/RESEARCH_GOAL_AND_PLAN.md).

**Current result:** O2 scored 906/1,533 (59.10%) with 8,192-token reasoning followed by a valid-letter readout, and 753/1,533 (49.12%) by ordinary answer generation. The original choice scorer gave 751/1,533 (48.99%). These are distinct protocols. V3 has not been trained or tested. The native Qwen comparison was stopped at the user's request and is not part of this reproduction path.

The folder includes only method source, a small Qwen module contract, scripts, and text documentation. External model weights and datasets must be downloaded under their respective access terms. Generated splits, caches, checkpoints, predictions, profiles, and logs are excluded. The full local historical O2 checkpoint is **not** in this folder; reproducing from scratch follows the training steps in the handoff.

## Layout

| Path | Purpose |
|---|---|
| `research/qwen35_omega/` | V1 split, feature cache, and single seed warm start |
| `research/qwen35_omega_v2/` | O2 split, fusion, training, O2-only evaluation, and V3 continuation |
| `reproduction/` | OmniSpatial prompt parser and B300 runtime setup |
| `reconstruction/` | Conda environment, relocation, Slurm budget wrapper, rationale join |
| `docs/` | Prior plan, small O2 metrics receipt, and dated research context |
| `results/.../native_interface_check.json` | 7.5 KB Qwen tensor and LoRA module contract required by the loader |

Before publishing, review the external models' and datasets' licenses. The gated VGGT-Ω checkpoint is deliberately absent.
