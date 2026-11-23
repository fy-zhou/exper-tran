# Result files and reproducibility settings

- **CrossVLAD:** [metrics and direction breakdown](crossvlad85/results.json), [per-image captions](crossvlad85/per_image.json), [cohort](crossvlad85/crossvlad_manifest.json), [attack budgets](crossvlad85/ATTACK_BUDGETS.md), [settings and BF16 audits](crossvlad85/audit/), and [image SHA-256 index](crossvlad85/image_artifact_index.csv).
- **LLaVA-Bench (COCO):** [quality decline](llava_bench_coco/QUALITY_DECLINE.md), [per-question rating pairs](llava_bench_coco/quality_per_question.csv), [clean and adversarial answers](llava_bench_coco/clean_adv_answers.jsonl), [frozen settings and BF16 audits](llava_bench_coco/settings/), [judge protocol](llava_bench_coco/judge/protocol.json), and [image SHA-256 index](llava_bench_coco/image_artifact_index.csv).
- **LLaVA matching:** [Ours per-question judgments](llava_bench_coco/ours_match_per_question.csv), [small experiments](llava_bench_coco/future_work/), and [future research](../../LLAVA_FINE_GRAINED_TRANSFER_PLAN.md).
- **ImageNet:** [prior full results and settings](../foa_anyattack_final_2026-11-17/) and [additional baseline results](../nov21_multitask/).

The snapshot contains text, structured measurements and settings. Image filenames and checksums identify the original local artifacts. Full PNGs, optimizer checkpoints, and raw judge traces remain on the permanent experiment volume at `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction`.
