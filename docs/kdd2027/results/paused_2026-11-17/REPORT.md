# Baseline snapshot at user pause — 2026-11-17 02:07 UTC

## Executive Summary

The 63-cell, seven-method × three-task × three-target campaign was paused at the user's request. Nine current formal ImageNet cells are complete (PGD50, MI-FGSM50, M-Attack300 × three targets). No new-task formal COCO/LLaVA evaluation is complete. Co-Attack has 120/150 frozen images and no formal target evaluation at this cutoff. Subsequent authorized Co/FOA continuation is outside this immutable snapshot.

IAU-FOA300's separate 10-image development pilot yielded Qwen27B 10/10, InternVL14B 10/10 and Gemma31B 8/10. These small-sample results do not establish full-test performance. Native five-proxy attacks are evaluated on Qwen/Gemma; InternVL uses the explicitly labeled four-proxy family-disjoint adaptation. Pilot attacks took approximately 4.0h and 2.65h respectively; serial full150 generation projects to about100h before further evaluation.

## Completed ImageNet results

All denominators below are150 fixed ground-truth-selected images. PGD/MI use an explicitly adapted target-image cosine objective, not classifier cross-entropy. No comparison below establishes superiority of Ours.

| Method | Steps | Qwen27B | InternVL14B | Gemma31B |
|---|---:|---:|---:|---:|
| PGD (adapted) |50|6/150 (4.0%)|7/150 (4.7%)|0/150 (0.0%)|
| MI-FGSM (adapted)|50|15/150 (10.0%)|14/150 (9.3%)|1/150 (0.7%)|
| M-Attack|50|63/150 (42.0%)|76/150 (50.7%)|44/150 (29.3%)|
| M-Attack|300|130/150 (86.7%)|135/150 (90.0%)|114/150 (76.0%)|

Budget differences must be shown in any comparison. The 50-step M results are additional historical controls, not three extra cells in the current63-cell table. See `historical_full_and_partial_results.csv` for per-direction rows, conditional clean-correct metrics, parser failures and Wilson intervals.

## Partial results and limitations

- Legacy FOA50:147/150 attacks frozen/evaluated; observed targeted hits Qwen76, InternVL88, Gemma63. Three missing outcomes remain unknown.
- Legacy FOA300:138/150 attacks frozen/evaluated; observed hits Qwen127, InternVL127, Gemma111. Twelve missing outcomes remain unknown. Neither observed-count ratio is a full150 ASR.
- Legacy FOA failures arose from incompatible/unpinned KMeans behavior. New FOA uses the compatible KMeans commit explicitly pinned by IAU-FOA (`f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`), supporting iteration limits and empty-cluster recovery. This is not claimed to be an exact FOA-author dependency pin. Revised FOA passed2-image smoke and requires an entirely separate full rerun; old and revised attacks cannot be mixed.
- Co-Attack:120 frozen formal images, no target scores; SGA:2-image smoke only. Their original joint retrieval algorithms are ported with auxiliary adversarial text preserved, but final image-only MLLM transfer is a task adaptation. Original untargeted retrieval objectives do not imply high ten-class targeted ASR.
- All seven current engineering smoke configurations passed: Co, SGA, revisedFOA, IAU native5, IAU family-disjoint4, COCO-caption pipeline and VQA pipeline. Initial formal-prefix debugging records are preserved separately; revised debugging cohorts are non-test. They are not final test results.
- COCO has150 fixed captioning images with concise caption prompt; LLaVA retains30 images/90 original open questions. Their metrics must not be renamed classification ASR. LLaVA official GPT4 judging is unconfigured: raw answers/reference ROUGE-L are diagnostics, not official correctness scores. No commercial calls were made.

## Evidence and validation

`structured_evidence.tar.gz` contains all JSON/JSONL/CSV/log/Markdown/YAML/text under the result root at cutoff: raw responses, configuration/source locks, image manifests, receipts, traces/losses, failed attempts and precision audits, including superseded engineering versions. `evidence_files.json` lists their hashes and sizes. Old versions, reused receipts and smoke runs are not independent observations and must not be pooled.

`receipt_integrity.csv` verifies1518 extension receipts across33 run/version directories (includes repeated/reused histories): every referenced clean/adv hash matches, decoded PNG L-infinity ≤16 integer levels, and receipt target-access flag is false. These receipt checks do not alone prove all aspects of algorithm correctness. All92 collected extension model-load audits report BF16-only parameters, unquantized and PASS. Exact parameter counts/revisions/roles are in `dtype_audits.json`; original audit evidence is in the archive. Previous49 unit/integration tests passed before launch.

Two clean/adversarial example pairs are selected by deterministic file order, not target success; receipts are included in `examples/`. They are examples, not additional test observations.

## Storage and recovery

Complete live checkpoints/images remain on persistent `/dev/vda1` at `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/`. `local_archive.json` identifies the full local output backup and SHA256. Canonical ImageNet inputs referenced from the previous V8 study also remain on persistent root; the output archive is not a standalone replacement for those externally referenced inputs. Only model/source-dataset cache is on `/ephemeral`.

The original broad queue remains paused. On2026-11-17 the user separately authorized ImageNet Co-Attack completion and corrected FOA completion only; `scripts/kdd2027/finish_foa_co.sh` implements that narrow continuation with original locks and per-image reuse. No new COCO/LLaVA or SGA/IAU formal work was launched by that continuation.
