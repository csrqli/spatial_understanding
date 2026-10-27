# OmniSpatial results survey — October 26, 2026

**Our completed O2 model is not established as state of the art.** Its full-test accuracy is **48.99% (751/1,533)**. More recent results include **54.2% with the same Qwen3.5-9B backbone**, and approximately **69% with larger models or agents**. Cross-paper comparisons below describe reported scores; prompts, training data, model sizes, and inference budgets differ.

## Current leaderboard versus recent papers

The [original OmniSpatial project leaderboard](https://qizekun.github.io/omnispatial/) still lists o3-2025-04-16 at **56.33%**. It is insufficient for judging the latest results. The authors also link to EASI. I retrieved the [public EASI API](https://easi.lmms-lab.com/api/leaderboard) on October 26; its response reports **lastUpdated = July 1, 2026, 17:34:33 UTC**. Thus even this live response is a dated snapshot. It contains 34 non-null OmniSpatial scores, with **Gemini 3 Pro at 69.14%** highest in that snapshot. [EASI leaderboard](https://easi.lmms-lab.com/leaderboard/), [API documentation](https://easi.lmms-lab.com/api-docs/).

| Model / method | Reported OmniSpatial accuracy | Our O2 minus reference (pp) | Evidence / qualification |
|---|---:|---:|---|
| Gemini 3 Pro | 69.14% | −20.15 | EASI API snapshot; manual-CoT score; larger proprietary model |
| SpatialBrain, GPT-5.6-sol agent | 69.1% | −20.11 | October 21, 2026 preprint, Table 1; explicitly evaluates all 1,533 questions |
| GPT-5.6-sol direct answers | 68.3% | −19.31 | Same SpatialBrain paper; shared-backbone baseline for that agent |
| SpatialClaw, Gemma4-31B | 63.6% | — | SpatialClaw Table 1; tools; fixed-seed 1,000-question subset |
| SpatialClaw, Qwen3.5-397B-A17B | 62.8% | — | Same subset protocol; larger open-weight MoE backbone and tools |
| o3-2025-04-16 | 56.33% | −7.34 | Original benchmark leaderboard; historical reference |
| SpatialOPSD, Qwen3.5-9B | 54.2% | −5.21 | October 8, 2026 preprint, OOD Table 2; same backbone family/size as ours |
| Qwen3.5-9B CoT baseline | 53.7% | −4.71 | Same SpatialOPSD table; generated reasoning baseline |
| **Our Ω-aligned Qwen3.5-9B, O2** | **48.99%** | **0.00** | All 1,533 questions; constrained answer-letter logits; one seed |
| Native pretrained Qwen3.5-9B, our matched choice evaluation | 47.03% | +1.96 | Newly measured on all 1,533 questions using the original scorer; 721 correct versus O2's 751; no generated reasoning |
| DCRL, Qwen3-VL-8B | 48.87% | +0.12 | Wide-baseline matching paper, Table 2; nearby model scale |
| SpatialCORE-8B | 48.46% | +0.53 | September 30, 2026 preprint, Table 1; official 1,533-question test |
| GeoThinker-F, Qwen3-VL-8B | 42.5% | +6.49 | October 14, 2026 preprint, EASI-ALL Table 13; this is the OmniSpatial column, not its multi-benchmark average |

Primary paper sources: [SpatialBrain Table 1](https://arxiv.org/html/2610.29317v1#S4.T1), [SpatialClaw Table 1](https://arxiv.org/html/2606.13673#S3.T1), [SpatialOPSD OOD table](https://arxiv.org/html/2610.11366v1#S4.T3), [DCRL Table 2](https://arxiv.org/html/2606.03577v1#S4.T2), [SpatialCORE Table 1](https://arxiv.org/html/2609.38716v1#S3.T1), and [GeoThinker-F Table 13](https://arxiv.org/html/2610.20592v1#A7.T13).

**These rows are not a unified leaderboard ranking.** The highest score verified in this survey is 69.14% in the retrieved EASI snapshot; this does not certify an exhaustive global record as of October 26. The conclusion that our current result does not establish SOTA also holds within the same 9B backbone scale: SpatialOPSD reports 54.2%.

**Coverage correction:** SpatialClaw caps benchmarks larger than 1,000 questions at a fixed-seed sample of 1,000. Its OmniSpatial entries therefore do not cover the complete 1,533-question test. The table marks their difference from our full-test score as unavailable. [SpatialClaw evaluation setup and Appendix B](https://arxiv.org/html/2606.13673#S4).

## Additional October work and implementation availability

[EvoSpatial, October 12](https://arxiv.org/html/2610.16679v1) reports **67.0%** with Gemma4-31B versus SpatialClaw's 63.6% and a no-tool 57.3% reference. It also uses a **1,000-question subset**, up to 30 tool-call turns, and an evolving skill/tool library during its main evaluation. These results are not a fixed-model, full-test comparison with our 9B system. Its HTML describes execution/evidence feedback without benchmark answer labels; the abstract-page text retrieved during the search describes supervised outcome feedback differently, so the exact feedback protocol should be checked against released code before reproduction. The linked [anonymous repository](https://anonymous.4open.science/r/anonymous-EvoSpatial-2533) exposes a public file listing and README, both retrieved locally. Its README states that failure reflection may consume answer feedback after evaluation, and warns that the example configurations do not certify reproduction of a paper table. This conflicts with the HTML’s label-free description; the precise main-result feedback setting remains unresolved. No datasets, checkpoints, trajectories or learned memories are included there. Its reported three-run aggregation is source information only; our own experiments remain seed 42 once.

[SpaceCast-Bench, October 8](https://arxiv.org/html/2610.12402v1#S4.SS4) provides a useful training-data direction: programmatically generated observe–transform–infer tasks, disjoint training scenes, and staged SFT with replay. Its **65.7% is on SpaceCast-Bench**, not OmniSpatial; its six OOD evaluations do not include OmniSpatial. It is not evidence that a 4B model exceeds 60% on our test. [Official code](https://github.com/ZJU-REAL/SpaceCast-Bench).

The following official repositories were downloaded under `third_party/research_survey/` and inspected rather than treated as interchangeable training packages:

| Repository | Checked commit | Implementation finding |
|---|---|---|
| [GCA](https://github.com/gca-spatial-reasoning/gca) | `4407cc08f330820b9ed7d13ff8ef81660e2d4107` | Reference-frame/objective workflow and geometric execution; example serving stack uses Qwen3-VL-235B-A22B-Thinking |
| [SpatialClaw](https://github.com/NVlabs/SpatialClaw) | `486b1b9925aea737de7e426fcad9af276cad2e14` | Stateful code agent, tool services and evaluators; no direct drop-in Ω-aligned Qwen3.5-9B trainer |
| [ReasonMatch / DCRL](https://github.com/aim-uofa/ReasonMatch) | `4d90eb2ee3567d075fdcc3b40b15b36651f9dc32` | veRL GRPO implementation, dynamic matching curriculum, geometric rewards; the release explicitly excludes training data |
| [GeoThinker / GeoThinker-F](https://github.com/Li-Hao-yuan/GeoThinker) | `061883c1df2b602a625d44588a9a8fc03b9af9e7` | Geometry fusion and predictive-head model code; Qwen2.5/Qwen3 stacks need adaptation for Qwen3.5 |
| [3DThinker](https://github.com/zhangquanchen/3DThinker) | `c9469e01b719310b0eaecc1133317e4ecfc74d8c` | Supervised geometry-latent alignment and outcome-based RL; its local MindCube trace export is usable after identity auditing |
| [SpatialCORE](https://github.com/rafiibnsultan/SpatialCORE) | `cfe0112e6a2383345c702c2175bf59f4f23ceb5c` | README only at the checked revision; training implementation unavailable |

No official SpatialOPSD implementation link was found in the checked paper and searches. No agent server, paid model endpoint, or additional training framework has been installed into the shared environment as part of this survey.

## Geometrically-Constrained Agent: promising, with different coverage

GCA formalizes the reference frame and objective, then executes geometric tools and Python. Its own OmniSpatial evaluation includes **Dynamic Reasoning and Perspective Taking only**; the published paper excludes Spatial Interaction and Complex Logic. [GCA Table 1 and Appendix D.3](https://arxiv.org/html/2511.22659v1#A4.SS3).

| GCA evaluation | Backbone baseline | GCA |
|---|---:|---:|
| Dynamic Reasoning | 60.5% | 73.6% |
| Perspective Taking | 43.9% | 58.6% |
| Combined two-category result | 51.0% | 65.1% |
| MMSI-Bench | 32.6% | 47.6% |
| MindCube-tiny | 47.3% | 64.2% |

Our two-category score is **49.24% (483/981)**, computed from the saved predictions. This aligns category coverage, not the entire evaluation protocol. GCA’s **65.1% is not a full four-category OmniSpatial score**. A later SpatialBrain table lists GCA at 63.0%; I exclude that entry from the main comparison because it is not GCA’s own original full-test result.

GCA’s reported deployment uses **eight A100 GPUs for the VLM and two for tools**, with multiple inference calls. That deployment is not a minimum hardware requirement, and the checked sections do not provide a matched per-question latency comparison. [GCA Appendix C.3](https://arxiv.org/html/2511.22659v1#A3.SS3).

## Implications for our method

1. **Do not claim overall or 9B-scale SOTA.** The small positive differences from DCRL and SpatialCORE do not establish superiority under unmatched protocols.
2. **Evaluate a matched native Qwen3.5-9B baseline and a generation/CoT protocol before attributing the gap to training.** The published native CoT score is numerically above ours, but comparing it with answer-letter logits cannot diagnose whether alignment helped or harmed the model.
3. **GCA is a useful baseline and possible teacher.** Our Ω alignment supplies geometry features; GCA adds explicit reference-frame specification and geometric execution. A future experiment could supervise reference-frame decisions or distill verified geometric traces while retaining the Ω alignment architecture. This is a proposed extension, not part of the completed O2 model.
4. **Measure inference cost alongside accuracy** when comparing feature alignment with tool-using agents. The current study has no matched agent-versus-O2 latency or compute experiment.

The subsequent improvement task includes a two-GPU matched generation diagnostic and a prepared rationale-supervised continuation. The diagnostic resumed under the new four-GPU **account-wide** cap; an unrelated job uses the other two GPUs. A CPU controller launches the single continuation only if the completed reasoning comparison remains below the target and the required account budget check passes. No follow-up training or external benchmark submission has yet occurred. The measured pretrained-native choice baseline is 47.03%. See the [active V3 improvement record](QWEN35_OMEGA_V3_IMPROVEMENT_PLAN_20261026.md) and the [downloaded related-work archive](related_work/README.md).

## Audit

- [Completed model metrics](results/qwen35_omega_omnispatial/20261025_scale_v2/test/O2_seed42/metrics.json)
- [Survey source snapshots, public EASI API response, and downloaded GCA CVPR paper](results/qwen35_omega_omnispatial/20261025_scale_v2/reports/omnispatial_latest_survey_sources/)
- [Meeting brief](QWEN35_OMEGA_MEETING_BRIEF_20261026.md)

Scores in the comparison table are question-level scores as reported by their respective sources, with sampled coverage explicitly marked. No multi-benchmark average, perspective-only result, or unsplit 8.4K OmniSpatial score is substituted for the full-test result. For sources without locally verified per-question predictions, coverage and protocol remain source-reported.
