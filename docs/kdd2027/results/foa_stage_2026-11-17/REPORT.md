# ImageNet baseline stage results

UTC cutoff: 2026-11-17T09:36:16Z

Status: **COMPLETE_FOA_AND_CO**. This completes only the requested ImageNet Co/FOA stage, not all seven methods or all three datasets.

|Method|Steps|Target|Hits / planned N|ASR|Status|
|---|---:|---|---:|---:|---|
|foa_attack|300|Gemma 4 31B-it|123/150|82.00%|COMPLETE|
|foa_attack|300|Qwen3.5-27B|138/150|92.00%|COMPLETE|
|foa_attack|300|InternVL3.5-14B-HF|138/150|92.00%|COMPLETE|
|co_attack|50|Gemma 4 31B-it|0/150|0.00%|COMPLETE|
|co_attack|50|Qwen3.5-27B|0/150|0.00%|COMPLETE|
|co_attack|50|InternVL3.5-14B-HF|0/150|0.00%|COMPLETE|

Corrected FOA replaces old partial FOA in the main comparison only if all150 attacks and all450 target rows validate. Original results remain archived; revised and old attacks are never mixed.

Co-Attack is supplemental image-only transfer from a joint untargeted retrieval method, excluded from the primary targeted comparison for threat-model mismatch, irrespective of its score. SGA is similarly excluded; neither is silently replaced.

PNG/receipt checks: 300; integrity PASS=True. Full raw replies, losses, configuration/source locks, revisions, audits, failures and previous exploratory records are in structured_evidence.tar.gz. No new GPU inference runs during reporting.

IAU remains pilot-only (Qwen10/10, InternVL10/10, Gemma8/10); V2/AnyAttack have separate authorized runs, excluded here. Missing scores remain N/A. Ours and old published artifacts are unchanged.

Persistent live output: `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction`. Full local image/checkpoint archive: `/home/ubuntu/exper-tran/outputs/archives/kdd2027_baselines_foa_stage_2026-11-17.tar.gz`. Canonical inputs in persistent V8 directories remain required. Model caches on /ephemeral are not backups.

The broad63-cell queue is paused. This watcher launches no experiment. Separately authorized live V2/AnyAttack files are excluded from this stable archive.
