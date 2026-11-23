# KDD2027 experiment results snapshot · 2026-11-23

## CrossVLAD 5 × 17

All 12 method-model conditions completed, totaling 1,020 evaluated image-method-model rows.

| Method | Steps | Qwen 27B | InternVL 14B | Gemma 31B | Overall |
|---|---:|---:|---:|---:|---:|
| Ours | 50 | 53/85 (62.4%) | 55/85 (64.7%) | 57/85 (67.1%) | 165/255 (64.7%) |
| FOA | 300 | 47/85 (55.3%) | 45/85 (52.9%) | 38/85 (44.7%) | 130/255 (51.0%) |
| M-Attack | 300 | 41/85 (48.2%) | 43/85 (50.6%) | 35/85 (41.2%) | 119/255 (46.7%) |
| SSA-CWA | 10 × 20 samples | 54/85 (63.5%) | 40/85 (47.1%) | 25/85 (29.4%) | 119/255 (46.7%) |

FOA uses 300 steps to avoid truncating that baseline to 50 steps. The methods use different attack budgets. See [budget details](crossvlad85/ATTACK_BUDGETS.md), [full metrics](crossvlad85/results.json), [per-image outputs](crossvlad85/per_image.json), and [generation times](crossvlad85/attack_runtime.csv).

## LLaVA-Bench (COCO)

[Full quality decline table and scorer details](llava_bench_coco/QUALITY_DECLINE.md). Every condition has 90 clean/adv question pairs. The following values are points of relative answer quality lost.

| Method | Qwen 27B | InternVL 14B | Gemma 31B |
|---|---:|---:|---:|
| Ours | 51.31 | 47.75 | 67.97 |
| M-Attack | 31.54 | 36.61 | 34.96 |
| FOA | 30.01 | 32.66 | 32.20 |
| MI-FGSM | 13.42 | 14.35 | 9.11 |
| PGD | 7.60 | 7.92 | 3.62 |

## ImageNet

- [150-image baseline and Ours table, including corrected FOA and AnyAttack](../foa_anyattack_final_2026-11-17/comparison_percent.csv).
- [SSA-CWA, AttackVLM variants, and completed GPT baseline evaluations](../nov21_multitask/REPORT.md).
- [Co-Attack and earlier pilot status](../foa_stage_2026-11-17/REPORT.md).

## Deferred scientific question

The LLaVA quality result establishes degradation under the tested perturbations. Fine-grained target matching, including color, position and relations, remains open. [Protocol and diagnostic results](llava_bench_coco/MATCH_PROTOCOL_AND_COMPARISON.md) and the [future experiment plan](../../LLAVA_FINE_GRAINED_TRANSFER_PLAN.md) document why this is separate. Frozen match records and smoke results are under the LLaVA directory.

Complete images, optimization traces and raw CLI logs remain on the persistent experiment volume. This snapshot contains tables, question/image-level measurements, manifests and precision audits.
