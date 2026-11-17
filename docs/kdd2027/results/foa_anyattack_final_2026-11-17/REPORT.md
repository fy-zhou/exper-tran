# Completed requested FOA + AnyAttack stage

UTC: 2026-11-17T09:41:22Z

Status: **COMPLETE_REQUESTED_SCOPE**.

The requested scope is corrected FOA150 and AnyAttack150, evaluated on three large targets. V2 was paused by the user after8 frozen images; IAU was not started. This is not completion of the historical63-cell campaign. No further experiment is launched.

|Method|Steps|Qwen27B|InternVL14B|Gemma31B|
|---|---:|---:|---:|---:|
|pgd|50|4.00%|4.67%|0.00%|
|mi_fgsm|50|10.00%|9.33%|0.67%|
|m_attack|50|42.00%|50.67%|29.33%|
|m_attack|300|86.67%|90.00%|76.00%|
|anyattack|0|0.67%|0.00%|0.00%|
|foa_attack|300|92.00%|92.00%|82.00%|
|Ours_final150_selected|50|91.33%|93.33%|89.33%|
|Ours_final150_FDA|50|91.33%|89.33%|92.67%|
|Ours_final150_uniform|50|92.00%|92.67%|91.33%|

All complete rates use all150 fixed ImageNet images. PGD/MI use a disclosed target-image cosine adaptation; AnyAttack is a released COCO-trained generator and has extra training resources. Native300 and matched50 budgets are distinct. Reference-image and proxy counts differ; this is not a fully compute/information-matched ranking. The new FOA fully validated run replaces old partial FOA in the main table; old evidence is preserved. Co/SGA remain supplemental/mismatched, not main targeted comparators.

V2 partial generation is preserved without an ASR; the raw controller may record its terminated worker as failed, but user_stop_v2.json establishes USER_PAUSED, not a scientific failure. No V2 resume or target evaluation was requested.

structured_evidence.tar.gz contains full raw answers, precision audits, model revisions, configs/source locks, loss traces, manifests, failures and intermediate exploratory evidence, including V2 partial receipts. PNG_integrity.json and BF16_integrity.json contain validation checks. Missing outcomes are not zeros.

Full images/checkpoints remain on persistent root: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction`. Full local archive: `/home/ubuntu/exper-tran/outputs/archives/kdd2027_FOA_AnyAttack_final_2026-11-17.tar.gz`. Canonical inputs in earlier persistent V8 directories remain required. Model weights on /ephemeral are disposable.
