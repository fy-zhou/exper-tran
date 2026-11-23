# CrossVLAD 17 × 5 attack budgets

| Method | Optimization budget for each image |
|---|---|
| Ours Uniform four layers | 50 updates |
| M-Attack | 300 updates |
| FOA, corrected | 300 updates |
| SSA-CWA with CLIP | 10 outer updates, 20 spectrum samples per update |

FOA uses 300 updates to avoid evaluating it under an optimization budget truncated to 50. The main table identifies this row as FOA-300. A fixed 50-step budget could understate FOA's performance. This campaign has no matched CrossVLAD FOA-50 row to quantify that difference.

Methods share the frozen source cohort and target evaluators, with different optimization budgets and reference inputs. Outer attack steps are not equivalent units of compute across methods. Report attack time per image alongside the step count and targeted hits out of 85. Archived ImageNet results and LLaVA's separate 50-step comparisons use their own budgets and must be kept distinct.
