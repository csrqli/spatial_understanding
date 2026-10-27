# V3: improve full-test OmniSpatial accuracy beyond 60%

Started October 26, 2026 at the user's request. This is an active follow-up to the completed O2 study, not a claim that the target has been reached.

## Target and fixed constraints

- Target: **more than 60% over all 1,533 official OmniSpatial test questions**, including Complex Logic and Spatial Interaction; at least 920 correct answers.
- Backbone: retain Qwen3.5-9B and the proposed frozen Ω spatial alignment unless the user changes that scope.
- At most **two concurrent GPUs for this task**, including evaluation and auxiliary feature workers. Exactly one training seed, **42**; no multi-seed repeats.
- Preserve completed V2 checkpoints and all earlier results. New artifacts go under `results/qwen35_omega_omnispatial/20261026_improve_v3/`.
- An observed test improvement is follow-up evidence: this public test has already been examined. Do not describe it as a newly blind test or select training examples using its answer labels.

## Evidence and implementation diagnosis

The [latest-result survey](OMNISPATIAL_LATEST_RESULTS_SURVEY_20261026.md) records original leaderboard scores, the retrieved EASI snapshot, recent papers, and differences in coverage. Our current 48.99% uses forced answer-letter scoring with thinking disabled. SpatialOPSD reports native Qwen3.5-9B CoT at 53.7% and its adapted model at 54.2%, under a different inference protocol. This motivates measuring a matched native baseline before attributing differences to our spatial alignment.

The current training instruction explicitly forbids explanations, and its target is only `Answer: X`. Training does not supervise reference-frame selection, intermediate spatial relations, or reasoning traces. Full-model training also includes a large SAT component dominated by binary spatial QA; transfer to abstract OmniSpatial logic is not established.

## First execution: matched reasoning evaluation

Implemented `research/qwen35_omega_v2/generation_compare.py` with two independent replicas: native pretrained Qwen3.5-9B and the completed Ω-aligned O2 checkpoint. Both receive the same images, prompt, decoding parameters, reasoning budget, and scoring rule. Frozen native and Ω features are reused from verified caches.

1. Engineering smoke on eight **training-side** examples; these are not independent accuracy estimates.
2. Fix response-budget or implementation failures using this engineering smoke before full test evaluation.
3. Freeze the protocol and evaluate native versus O2 on the complete official test once each. Also measure native under the original choice-scoring protocol to distinguish adaptation from inference changes.
4. Report freely generated answer accuracy and reasoning-conditioned option scoring separately. The latter always closes the thinking block and scores valid letters after a fixed `Answer:` prefix; it is applied to all items and never chooses between predictions using correctness.

The first 2,048-token smoke completed but truncated six of eight responses for each model. This is a response-budget failure, not a useful benchmark accuracy estimate. A subsequent engineering job uses an 8,192-token budget and the declared reasoning-conditioned readout.

Initial jobs: **8901** (completed 2K engineering smoke); **8904** (8K engineering smoke). Both request two idle RTX PRO 6000 Blackwell GPUs, 64 CPUs and 320 GiB host RAM while B300 GPUs are occupied. No completed test answer is overwritten or regenerated to choose a more favorable sample.

The 8K smoke passed. Full job **8905** was cancelled externally by UID 0 without a recorded reason. The same-resource retry **8906** and CPU-only report monitor **8908** were both cancelled by UID 0 at **14:18:12 SGT**. Slurm records no reason. The user then clarified an account-wide four-GPU cap, and another account job finished, leaving two GPUs available. Job **8911** resumed the same immutable protocol on two B300s at **16:09 SGT**; the remaining two account GPUs are used by an unrelated project. Its two models restore their own saved predictions and RNG states. CPU monitor **8912** reports exact coverage. No task exceeds two simultaneous GPUs.

### First completed matched result

Native pretrained Qwen3.5-9B, using the exact original answer-letter scoring function, scores **47.03% (721/1,533)**. Completed O2 scored **48.99% (751/1,533)** under that protocol, a **+1.96 percentage-point** difference. This compares the full adapted model with pretrained Qwen; it does not isolate Ω alignment from merger/LoRA adaptation or training data.

| Main task | Pretrained native | Completed O2 |
|---|---:|---:|
| Complex Logic | 23.41% | 29.76% |
| Dynamic Reasoning | 49.29% | 50.95% |
| Perspective Taking | 47.42% | 47.95% |
| Spatial Interaction | 63.00% | 64.33% |

Artifact: `results/qwen35_omega_omnispatial/20261026_improve_v3/locked_reasoning_8k/native_choice/metrics.json`. At the interruption, **64/1,533 native and 96/1,533 O2 predictions** had been preserved. Job 8911 is extending those outputs; the live counts are in `locked_reasoning_8k/comparison_progress.json`. Until both models cover all 1,533 questions, there is no complete generated-reasoning score or verified >60% result.

### Available reasoning supervision

The local `3DThinker-10K/data_output3d_begin_10k_resized.jsonl` contains 10,000 reasoning traces for the existing MindCube training questions. I matched **all 10,000** to the raw MindCube release using exact normalized question text, ordered source image filenames, and the answer letter. The average reasoning length is approximately 1,627 characters. The traces' annotations and identities match; every intermediate geometric claim has not been independently verified.

Prepared annotation file: `results/qwen35_omega_omnispatial/20261026_improve_v3/data/verified_mind_reasoning.jsonl`; source hashes and checks are in `data/reasoning_audit.json`. These add richer supervision, **not 10,000 additional unique questions**. The integer `idx` is not the raw MindCube row index; direct index matching would silently pair the wrong images/questions, so it is not used.

## Paper and code survey

| Work | Relevant evidence | Code status checked October 26 |
|---|---|---|
| GCA | Explicit reference-frame/objective constraints and geometric execution; its original 65.1% OmniSpatial result covers only two categories | Official repository cloned and inspected: `third_party/research_survey/gca`, commit `4407cc08f330820b9ed7d13ff8ef81660e2d4107` |
| SpatialClaw | Code-based spatial tool use; matched native/agent comparisons across Qwen3.5 and other backbones | Official repository cloned and inspected: `third_party/research_survey/SpatialClaw`, commit `486b1b9925aea737de7e426fcad9af276cad2e14` |
| SpatialOPSD | Same 9B backbone; uses verified spatial-agent traces as privileged distillation context | Paper and appendix checked; no official implementation link found in the paper or searches so far |
| SpatialCORE | Confidence-weighted grounding rewards; 48.46% with an 8B backbone | Official repository cloned, commit `cfe0112e6a2383345c702c2175bf59f4f23ceb5c`; the checked revision contains a README but no training implementation |
| DCRL | Correspondence-based RL; 48.87% with an 8B backbone | Cloned and inspected `third_party/research_survey/ReasonMatch`, commit `4d90eb2ee3567d075fdcc3b40b15b36651f9dc32`; training code available, training data not released there |
| GeoThinker-F | Distills geometry representations into the VLM; removes the external encoder at inference | Cloned and inspected `third_party/research_survey/GeoThinker`, commit `061883c1df2b602a625d44588a9a8fc03b9af9e7`; existing code targets Qwen2.5/Qwen3 |
| 3DThinker | Geometry-latent alignment and reasoning/outcome supervision | Cloned and inspected `third_party/research_survey/3DThinker`, commit `c9469e01b719310b0eaecc1133317e4ecfc74d8c`; local 10K MindCube traces audited and tokenized |

Sources: [GCA code](https://github.com/gca-spatial-reasoning/gca), [SpatialClaw code](https://github.com/NVlabs/SpatialClaw), [SpatialCORE code](https://github.com/rafiibnsultan/SpatialCORE), [SpatialOPSD paper](https://arxiv.org/abs/2610.11366), [DCRL code](https://github.com/aim-uofa/ReasonMatch).

The inspected GCA evaluator can load all categories, but the original paper evaluates only Dynamic Reasoning and Perspective Taking. Its evaluator also has an `ignore_empty` option; our full-test scorer retains every official question and counts invalid/missing answers as incorrect. SpatialClaw's paper uses a 1,000-question subset, and its evaluator defaults unparseable answers to A; our diagnostic instead records invalid answers explicitly. These details matter for comparisons.

## Next method changes to assess after the diagnostic

- Preserve native reasoning capability while training alignment: reasoning supervision and/or distillation, rather than answer-only language adaptation.
- Make reference frames and geometric objectives explicit, borrowing the useful GCA structure while retaining Ω feature alignment.
- Adjust the training mixture toward OmniSpatial's dynamic, viewpoint and abstract-logic tasks; compare on development data rather than iterating on test answers.
- Where useful, distill verified spatial-tool traces into the 9B model. A tool-using agent is a separate inference-cost setting and must be labeled accordingly.

Only the best justified final candidate should receive a new full training run. This list is an investigation plan, not a queue of baseline/ablation training jobs. Exceeding 60% is a research target, not a guaranteed outcome.

## Prepared single continuation: reasoning-supervised O2

Implemented `research/qwen35_omega_v2/rationale_training.py`; **not launched or GPU-verified**. It initializes from the completed O2 checkpoint and preserves both frozen encoders, local alignment, scene context, merger and LoRA architecture. Native answer-only examples retain their existing prompt mode; MindCube rationale variants use the reasoning-enabled prompt. The original language-model weights remain frozen.

The previous mixture contains **106,033 SAT `other` training questions**, versus only **59 OmniSpatial Dynamic Reasoning training questions**. More total examples did not provide balanced dynamic or abstract-logic supervision. The prepared continuation uses:

| Source | Unique training questions | Exposures per epoch |
|---|---:|---:|
| OmniSpatial training partition | 5,172 | 12,706 |
| SAT, up to 1,600 questions in each of six task types | 9,600 | 9,600 |
| MindCube with matched 3DThinker reasoning | 9,019 | 18,038 |
| **Total** | **23,791** | **40,344** |

Omni replay weights are Interaction 2, Perspective 2, Complex Logic 4 and Dynamic Reasoning 12. MindCube reasoning examples repeat twice; SAT examples once. The smaller unique mixture concentrates supervision rather than adding duplicate questions as new data. Two epochs provide **80,688 exposures, 2,522 optimizer updates**, and approximately **14.31 million supervised tokens**. These are planned counts, not completed training.

The objective gives 50% weight to the final answer and 50% to per-question mean reasoning/format cross entropy. It identifies the final answer by an explicit token position, so option letters mentioned inside reasoning are not mislabeled as final answers. CPU checks passed for loss weighting, repeated option letters, invalid answer positions and finite gradients. The token preparation accepted all 10,000 source traces, with no truncation; maximum prepared sequence length is 4,886 tokens.

Global batch remains 32 across two GPUs. Peak learning rates are alignment `2e-5`, visual merger `1e-6`, and LoRA `1e-6`, with 3% warmup, cosine decay and gradient clipping at 1. These are conservative continuation settings, not empirically proven optimal parameters. The first real optimizer steps must verify GPU memory and throughput before continuing the same saved optimizer/RNG state. Source and data hashes prevent an accidental change of objective or batching during resume.

Audit artifacts: `data/focused_recipe_audit.json`, `data/reasoning_audit.json`, `data/reasoning_manifest.jsonl`, `data/reasoning_tokens.pt`, and `implementation_verification.json` under the V3 result directory. Available HSTAR SFT files were inspected, but their robot navigation/action trajectories are not directly comparable to these spatial multiple-choice questions; they have not been mixed into this candidate.

## Active single-model follow-up

CPU controller job **8918** monitors job 8911 and then follows a locked decision rule. If completed O2 exceeds 60% on all 1,533 questions with reasoning-conditioned readout, it records the inference result and starts no new training. Otherwise it runs two genuine optimizer steps as a GPU preflight, resumes those exact weights/optimizer/RNG states to finish **one** seed-42 rationale continuation, and evaluates that final checkpoint on all 1,533 questions. Two GPU shards share the final evaluation; their per-question outputs are checked and merged before any overall accuracy is reported. There is no multi-seed or ablation queue.

Before **each** future GPU submission, the controller runs `bash ~/.claude/skills/xaai-cluster/scripts/gpu_budget.sh <GPUS_REQUESTED>` and submits only after `OK`. It counts running and pending account GPUs. If the budget check refuses, a job is cancelled, or validation fails, the controller stops and records the reason rather than retrying. The current approved cap is four account GPUs through 16 November 2026, reverting to two from 17 November unless support changes it sooner. This task itself still uses at most two concurrent GPUs.

The prior cancellation audit is preserved as `locked_reasoning_8k/interrupted_before_8911.json`; the live comparison report is `locked_reasoning_8k/COMPARISON_REPORT.md`. At least **920/1,533 correct** is required to exceed 60%, which is 169 more correct answers than completed O2's original protocol. The local [related-work archive](related_work/README.md) contains the papers, released code and full-versus-subset comparison audit.
