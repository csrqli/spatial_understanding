# Rebuild the Qwen3.5–VGGT-Ω OmniSpatial method in a new account

This is the source-to-result handoff for the **current O2 model** and the **planned V3 continuation**, updated 2026-10-27. **The research goal is OmniSpatial SOTA on all 1,533 official questions.** The earlier idea and plan are in [RESEARCH_GOAL_AND_PLAN.md](docs/RESEARCH_GOAL_AND_PLAN.md), with dated originals in [prior_plans](docs/prior_plans/). O2 is Qwen3.5-9B plus frozen VGGT-Ω features, trained with the single seed 42. The rationale-supervised V3 continuation has code and prepared training-side data, but has **not been trained or evaluated**. This handoff runs O2 and V3 only; the redundant native Qwen comparison is deliberately omitted.

The custom alignment, cache, split, training, and evaluation source is **included in this folder**; it is intended for your later Git repository. Upstream repositories alone do not implement O2 or V3. No previous O2 checkpoint, feature cache, or private annotation is required if all stages below are run. VGGT-Ω weights are publicly listed but access gated under a noncommercial research license, so each account must obtain its own approved access and follow the model and dataset terms. This folder intentionally contains no model weights, dataset files, tensor caches, checkpoints, full prediction files, logs, or other large generated artifacts.

## Public inputs and fixed identities

| Input | Creator release and use | Identity in current run |
|---|---|---|
| Qwen3.5-9B | [Qwen model](https://huggingface.co/Qwen/Qwen3.5-9B), Apache-2.0 | revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| VGGT-Ω code | [Meta GitHub](https://github.com/facebookresearch/vggt-omega) | local checkout `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` |
| VGGT-Ω weights | [Meta checkpoint](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_416_reproduce.pt), [reproduction note](https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md) | `vggt_omega_1b_416_reproduce.pt`, SHA256 `e900833330eb0e570e10543e6108d0dd704fa7c8dbc32738a645987875298896`; access gated, FAIR Noncommercial Research License |
| OmniSpatial train/test | [official dataset](https://huggingface.co/datasets/qizekun/OmniSpatial), [code](https://github.com/qizekun/OmniSpatial) | dataset revision `6691f3288bb1ff207d6ead4d841b505de08a6fd8`; use `OmniSpatial-train.zip`, `OmniSpatial-test.zip` |
| SAT-v2 | [official dataset](https://huggingface.co/datasets/array/SAT-v2), [code](https://github.com/arijitray1993/SAT) | revision `77d83da806d0a31f837be7b36d0dfbda230d8864`; train, val and test Parquet files |
| MindCube | [official dataset](https://huggingface.co/datasets/MLL-Lab/MindCube), [code](https://github.com/mll-lab-nu/MindCube) | dataset revision `9c941b46a6bd65b6914669ef7a579948fc9c8467`; `data.zip` |
| 3DThinker rationales | [author dataset](https://huggingface.co/datasets/jankin123/3DThinker-10K) | revision `2b16e1e73cf985e5d46b84cc90c13956bc7205f2`; `data_output3d_begin_10k_resized.jsonl`; CC-BY-NC-4.0 |

The 3DThinker questions duplicate MindCube training questions; they supply rationale text, **not** 10,000 extra unique questions. The local rationale join checks normalized question text, *ordered full relative image paths*, and answer letter. The dataset's integer `idx` is not a MindCube row index. Teacher text has not been checked for geometric truth question by question.

## New folder and environment

Copy **this entire folder** to a writable absolute path in the new account and set `PROJECT` to that path. Its `results/` tree contains only a small Qwen interface contract, not generated results. Run relocation on the copied folder before preparing data or training. The `research/vlm3r_omega/vendor/vggt-omega` checkout is downloaded separately.

```bash
export PROJECT=/absolute/path/in/new/account/omnispatial_o2_v3_release
cd "$PROJECT"
mkdir -p datasets models eval_logs
conda env create -p "$PROJECT/env" -f "$PROJECT/reconstruction/environment.yml"
conda activate "$PROJECT/env"
unset PYTHONPATH
bash "$PROJECT/reconstruction/install_b300_overlay.sh"
```

The YAML pins Python 3.10, PyTorch 2.9.0 CUDA 12.8, Transformers 5.20.2, FLA 0.5.2, PEFT 0.21.2, and the data stack used by the method. The separate overlay command pins the exact Triton 3.7.0 and cuDNN 9.17.0.29 versions used on B300; these deliberately override the Torch wheel's declared Triton 3.5.0 and cuDNN 9.10.2.21 requirements, so `pip check` will report those two mismatches. The exact overlay was validated in the existing run, but it is not a generally supported package combination. The YAML omits vLLM, FlashInfer and ordinary VGGT packages because the final O2/V3 path does not import them. `flash_attn` was present in the old shared overlay but O2 uses SDPA and the pinned FLA gated-delta kernels. For B300 (sm_103) on this cluster, the Slurm wrapper also needs the installed CUDA 12.9 toolkit/NVRTC path from `reproduction/slurm_b300_env.sh`; the YAML cannot install cluster drivers or Slurm.

Download the public sources under the new account (Hugging Face CLI comes with `huggingface_hub`). Authenticate and accept the VGGT-Ω access conditions for that account before downloading its weights.

```bash
cd "$PROJECT"
hf auth login
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --local-dir "$PROJECT/models/Qwen3.5-9B"
mkdir -p research/vlm3r_omega/vendor
git clone https://github.com/facebookresearch/vggt-omega.git research/vlm3r_omega/vendor/vggt-omega
git -C research/vlm3r_omega/vendor/vggt-omega checkout 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
hf download facebook/VGGT-Omega vggt_omega_1b_416_reproduce.pt --local-dir "$PROJECT/models/vggt-omega"
sha256sum "$PROJECT/models/vggt-omega/vggt_omega_1b_416_reproduce.pt"
hf download qizekun/OmniSpatial OmniSpatial-train.zip OmniSpatial-test.zip --repo-type dataset --revision 6691f3288bb1ff207d6ead4d841b505de08a6fd8 --local-dir "$PROJECT/datasets/omnispatial"
hf download array/SAT-v2 --repo-type dataset --revision 77d83da806d0a31f837be7b36d0dfbda230d8864 --local-dir "$PROJECT/datasets/spatial_alignment_v2/SAT-v2"
hf download MLL-Lab/MindCube data.zip --repo-type dataset --revision 9c941b46a6bd65b6914669ef7a579948fc9c8467 --local-dir "$PROJECT/datasets/3dthinker/MindCube"
hf download jankin123/3DThinker-10K data_output3d_begin_10k_resized.jsonl --repo-type dataset --revision 2b16e1e73cf985e5d46b84cc90c13956bc7205f2 --local-dir "$PROJECT/datasets/3dthinker/3DThinker-10K"
unzip -q "$PROJECT/datasets/omnispatial/OmniSpatial-train.zip" -d "$PROJECT/datasets/omnispatial/extracted/train"
unzip -q "$PROJECT/datasets/omnispatial/OmniSpatial-test.zip" -d "$PROJECT/datasets/omnispatial/extracted/test"
unzip -q "$PROJECT/datasets/3dthinker/MindCube/data.zip" -d "$PROJECT/datasets/3dthinker/MindCube/extracted"
python reconstruction/relocate.py --project "$PROJECT" --env-prefix "$PROJECT/env"
```

Check that `sha256sum` printed the Ω hash above and that these exist: `datasets/omnispatial/extracted/train/OmniSpatial-train/data.json`, `datasets/omnispatial/extracted/test/OmniSpatial-test/data.json`, `datasets/3dthinker/MindCube/extracted/data/raw/MindCube_train.jsonl`, and `datasets/spatial_alignment_v2/SAT-v2/data/train-00000-of-00011.parquet`. The relocation script rewrites only the destination method source and Slurm wrappers. Run it before making any cache or checkpoint, since saved configurations lock code hashes and paths. Keep the included `results/qwen35_omega_omnispatial/20261024_revision/manifests/native_interface_check.json`: it records the Qwen tensor and LoRA module contract and is required at model load. If the cluster CUDA toolkit is at a different path, change `CUDA_HOME` in `reproduction/slurm_b300_env.sh` and the `LD_PRELOAD` line in `reconstruction/run_gpu.sbatch`, then verify the GPU preflight.

## Recreate the model, in order

Run every Python command from `$PROJECT` using `$PROJECT/env/bin/python`, with `PYTHONPATH` unset except inside the Slurm wrapper. CPU preparation runs on a login/CPU node and GPU commands run inside Slurm allocations. `reconstruction/submit_gpu.sh` invokes the account-wide budget checker immediately before each GPU `sbatch` and rechecks after submission. These commands use **one seed, 42**, per setting. Keep this project within two concurrent GPUs, including feature extraction and evaluation. On this XAAI account the cap is four GPUs through 2026-11-16 and two thereafter, subject to any administrator reduction. The checker counts all running and pending jobs. Do not cancel unrelated jobs. On another cluster, replace the checker with that cluster's account-wide quota policy; do not bypass the check.

1. **Create the V1 OmniSpatial-only split and common warm start.** `research.qwen35_omega.data` makes image-content groups with split seed `20261024`, reading test *pixels* only to reject exact overlap, never test labels. Target output is `results/qwen35_omega_omnispatial/20261024_revision/splits/content_groups`. Run `research.qwen35_omega.cache --kind tokens`, then `--kind native` on a GPU. Check both `*_complete.json` receipts. Train only `B1` seed 42 for four epochs, Stage A one epoch and Stage B three, effective batch 32. The target checkpoint is `.../pilot/B1_seed42/epoch_4.pt`. B1 here is the required initialization for O2, not a new experimental baseline study.

   ```bash
   export V1="$PROJECT/results/qwen35_omega_omnispatial/20261024_revision"
   "$PROJECT/env/bin/python" -m research.qwen35_omega.data --train-root "$PROJECT/datasets/omnispatial/extracted/train/OmniSpatial-train" --test-image-root "$PROJECT/datasets/omnispatial/extracted/test/OmniSpatial-test" --output "$V1/splits/content_groups" --seed 20261024
   "$PROJECT/env/bin/python" -m research.qwen35_omega.cache --kind tokens
   bash reconstruction/submit_gpu.sh 1 q35_v1_native research.qwen35_omega.cache --kind native --batch-size 32
   # Wait for the native cache job to finish successfully.
   bash reconstruction/submit_gpu.sh 1 q35_v1_warm research.qwen35_omega.train --arm B1 --seed 42 --split train --microbatch 16 --short-microbatch 32 --output "$V1/pilot/B1_seed42"
   ```

2. **Build the V2 train-side mixture.** `research.qwen35_omega_v2.prepare` merges OmniSpatial, SAT-v2 and MindCube while preserving image order; it removes exact decoded-pixel overlap and duplicate QA, keeps linked images/scene directories in one partition, excludes MindCube tinybench scenes, and never reads held-out test labels. The SAT choices are deterministically shuffled. It writes `train`, `development`, `confirmation`, `full_train`, `pilot_train`, `images`, and `sequences` under `.../20261025_scale_v2/splits`. Expect **168,520 unique** `full_train` questions: 151,623 SAT, 6,897 OmniSpatial and 10,000 MindCube. Do not add 3DThinker-10K as extra data. Generate V2 `tokens`, `native`, and `omega` caches; do not compute `vggt` for the final model. The cache identity pins the Qwen revision, Ω SHA256, image pixel bounds 65,536–1,003,520, and white square letterboxing at 416 pixels. Check all cache completion receipts, then run `research.qwen35_omega_v2.preflight` on a GPU.

   ```bash
   export V2="$PROJECT/results/qwen35_omega_omnispatial/20261025_scale_v2"
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.prepare
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.cache --kind tokens --subset full_train
   bash reconstruction/submit_gpu.sh 1 q35_v2_native research.qwen35_omega_v2.cache --kind native --subset full_train --batch-size 32
   bash reconstruction/submit_gpu.sh 1 q35_v2_omega research.qwen35_omega_v2.cache --kind omega --subset full_train --batch-size 128
   # Wait for both caches to finish successfully.
   bash reconstruction/submit_gpu.sh 1 q35_v2_preflight research.qwen35_omega_v2.preflight
   ```

3. **Train final O2.** The frozen Qwen vision encoder and frozen Ω aggregator feed a 512-wide residual alignment module. Each Qwen pre-merger visual token attends to a 14×14 Ω local grid plus camera token, and separately to one Ω camera plus 16 scene registers. LayerNorm and zero-initialized output projections preserve the native visual path at initialization. Stage A trains only alignment for one epoch. Stage B trains alignment, Qwen visual merger, and rank-16/alpha-32/dropout-0.05 language LoRA for three epochs; other base weights stay frozen. Loss weights final answer letter and response-format tokens equally. Data replay per epoch is SAT ×1, OmniSpatial ×4, MindCube ×2: **199,211 exposures/epoch**, four epochs, **24,904** optimizer updates, global batch 32, seed 42. The final checkpoint is `.../full/O2_seed42/epoch_4.pt`. This clean one-GPU command preserves the method and global batch; the historical run changed from one to two data-parallel GPUs during continuation, so a clean run is not bitwise identical.

   The completed study fixed the four-epoch horizon using an earlier B1 pilot. For a final-model-only rebuild, record that historical design choice instead of rerunning the deferred B1 V2 pilot. Write this lock **before** O2 training; the test-preparation code verifies O2 completion before reading test labels.

   ```bash
   mkdir -p "$V2/manifests"
   "$PROJECT/env/bin/python" -c 'import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"schema":"qwen35_omega_v2_locked_design","full_arms":["O2"],"seeds":[42],"selected_pilot_total_epoch":4,"joint_epochs":3,"selection_arm":"historical_B1_pilot","test_labels_opened":False},indent=2)+"\n")' "$V2/manifests/design_lock.json"
   # Wait for preflight, then submit the full refit.
   bash reconstruction/submit_gpu.sh 1 q35_v2_o2_full research.qwen35_omega_v2.train --arm O2 --seed 42 --subset full_train --joint-epochs 3 --output "$V2/full/O2_seed42"
   ```

4. **Prepare the V3 candidate inputs.** Run the provided public-data join; it must report 10,000 exact matches and source SHA256 values `d0ee3f89...` for MindCube and `027caf76...` for 3DThinker. `prepare_reasoning` tokenizes the matched rationales and checks that each answer is the exact final letter. Its training partition has **23,791 unique** questions: 5,172 OmniSpatial, 9,600 SAT, 9,019 MindCube reasoning; **40,344 exposures/epoch**. The two-epoch continuation has **2,522 updates** with global batch 32 across two GPUs; objective is 0.5 final-answer CE plus 0.5 per-question mean reasoning/format CE, fusion LR `2e-5`, merger and LoRA LR `1e-6`. The data preparation and loss self-check use no GPU.

   ```bash
   export V3="$PROJECT/results/qwen35_omega_omnispatial/20261026_improve_v3"
   "$PROJECT/env/bin/python" reconstruction/match_mindcube_reasoning.py --mindcube-train "$PROJECT/datasets/3dthinker/MindCube/extracted/data/raw/MindCube_train.jsonl" --thinker-jsonl "$PROJECT/datasets/3dthinker/3DThinker-10K/data_output3d_begin_10k_resized.jsonl" --output "$V3/data/verified_mind_reasoning.jsonl"
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.prepare_reasoning
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.rationale_training --check-loss
   ```

5. **Evaluate O2 on the full official OmniSpatial test.** Once O2 training and design lock are recorded, `prepare_test` opens the held-out labels. Build test token/native/Ω caches. `evaluate` provides the original fixed `Answer:` valid-letter choice score. `generation_compare --single-arm O2` evaluates **only our completed O2 model** on one GPU: one seed-42 rollout per question, up to 8,192 generated tokens, temperature 1, top-p .95, top-k 20, presence penalty 1.5. It records free-generation accuracy and a separate ground-truth-independent reasoning-conditioned valid-letter readout. `finalize_aligned` verifies **all 1,533 unique IDs** and reports invalid answers, token-cap hits, and per-category scores. The test was already seen during method development, so this is follow-up evidence rather than a new blind estimate. Do not present the readout score as ordinary free-generation accuracy.

   ```bash
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.prepare_test
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.cache --kind tokens --subset test
   bash reconstruction/submit_gpu.sh 1 q35_test_native research.qwen35_omega_v2.cache --kind native --subset test --batch-size 32
   bash reconstruction/submit_gpu.sh 1 q35_test_omega research.qwen35_omega_v2.cache --kind omega --subset test --batch-size 128
   # Wait for both test caches to finish.
   bash reconstruction/submit_gpu.sh 1 q35_o2_choice research.qwen35_omega_v2.evaluate --arm O2 --checkpoint "$V2/full/O2_seed42/epoch_4.pt" --split test --output "$V2/test/O2_seed42"
   bash reconstruction/submit_gpu.sh 1 q35_o2_reasoning research.qwen35_omega_v2.generation_compare --output "$V3/locked_reasoning_8k" --checkpoint "$V2/full/O2_seed42/epoch_4.pt" --batch-size 32 --max-new-tokens 8192 --reason-then-choice --single-arm O2
   # After Slurm reports successful completion, verify exactly 1,533 IDs:
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.finalize_aligned --output "$V3/locked_reasoning_8k"
   ```

6. **Continue and evaluate V3 if O2 remains below the research goal.** The completed O2 result is 906/1,533 under the locked readout, so the planned rationale continuation is warranted. Run `--check-loss` as above and a short *resumable segment of the same seed-42 run* before the full continuation. The probe and full job must use the same output, optimizer state, and two GPUs. Do not start an independent smoke checkpoint or change the recipe mid-run. Verify the `complete.json` receipt says seed 42 and 2,522 global steps. Use the exact O2 diagnostic settings for final V3 evaluation; `finalize_aligned` rejects missing or duplicate official IDs and merges the two shards.

   ```bash
   RECON_DDP=1 bash reconstruction/submit_gpu.sh 2 q35_v3_probe research.qwen35_omega_v2.rationale_training --output "$V3/rationale_O2_seed42" --initialize "$V2/full/O2_seed42/epoch_4.pt" --epochs 2 --max-steps 2
   # Wait for the probe and inspect its finite losses/peak memory.
   RECON_DDP=1 bash reconstruction/submit_gpu.sh 2 q35_v3_full research.qwen35_omega_v2.rationale_training --output "$V3/rationale_O2_seed42" --initialize "$V2/full/O2_seed42/epoch_4.pt" --epochs 2
   # Wait for training completion.
   RECON_DDP=1 bash reconstruction/submit_gpu.sh 2 q35_v3_final research.qwen35_omega_v2.generation_compare --output "$V3/final_reasoning_8k" --checkpoint "$V3/rationale_O2_seed42/epoch_2.pt" --batch-size 32 --max-new-tokens 8192 --reason-then-choice --only-aligned
   # After both final shards finish:
   "$PROJECT/env/bin/python" -m research.qwen35_omega_v2.finalize_aligned --output "$V3/final_reasoning_8k"
   ```

The current O2 full-test choice score was **751/1,533 = 48.99%**; its later reasoning-conditioned readout was **906/1,533 = 59.10%**, and ordinary generated-answer accuracy was **753/1,533 = 49.12%**. These are existing receipts, not promised results for a clean rebuild. The V3 training and evaluation have not run. Report whichever completed O2/V3 model actually wins under the same full-test protocol. Aim for `>=920/1533` to exceed 60%; `>=1060/1533` exceeds the surveyed 69.14% reference, subject to protocol comparability.

## Verify before reporting

- Confirm the Qwen revision and Ω SHA256 before extraction; confirm `config.py` points into the new project.
- Preserve `splits/manifest.json`, cache identity receipts, `configuration.json`, `complete.json`, source hashes, checkpoint names, and per-question predictions. Do not copy a previous account's cached tensors into a new source tree with different path/code identity.
- Run `python -m unittest research.qwen35_omega.test_contracts research.qwen35_omega_v2.test_contracts -v`, then the GPU preflight. Check `torch.version.cuda`, the allocated GPU model, and the CUDA 12.9 NVRTC preload on B300.
- This curated folder excludes historical multi-seed and ablation launchers. Run only seed 42 for O2 and its optional V3 continuation; keep baseline and ablation training deferred.
