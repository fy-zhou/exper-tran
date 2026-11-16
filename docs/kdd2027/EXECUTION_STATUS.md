# Live campaign status — 2026-11-16 11:29 UTC

**RUNNING, not completed.** Full campaign launched in independent tmux session
`kdd2027-baselines`; detached from the Codex terminal. No commercial calls.

Verified before launch:

- 33 unit/integration checks pass, Ruff clean.
- Three-proxy, four-method, two-image GPU smoke:8 valid bounded PNGs.
- Four-method screen10 ×50 steps:40 valid PNGs, no failed attacks.
- Qwen27B, InternVL14B, Gemma31B end-to-end smoke:24 adversarial evaluation rows;
  all three complete models recursively audited BF16/unquantized.
- Model weights restored at pinned revisions: approximately145GiB on
  `/ephemeral/kdd2027/hf-cache/` (filesystem `/dev/vdb`, temporary).
- All result/receipt/log paths on root `/dev/vda1`, under
  `/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/`.
- Previous Ours9×150 attacks validated read-only; no Ours/paper/old-result edits.

Execution order: matched50 PGD/MI/M/FOA ×150images, then target evaluation on
three large models; original300 M/FOA ×150images and their evaluations. The50
and300 budgets have distinct immutable locks and output directories. Baseline
loss/precision/task adaptations are explicit; neither study claims the original
papers' free-form caption metrics or commercial results.

Pilot attack-only times for10 images: PGD54.2s; MI53.3s; M94.7s; FOA136.4s.
Projection from these measurements gives ~85minutes for matched50 attacks and
~5.8hours for native300 attacks, plus target evaluations/model loads. This is a
projection, not a verified completion promise; native300 still needs measurement.

Monitor actual progress:

```bash
tmux capture-pane -p -t kdd2027-baselines -S -30
tail -n 20 /home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/logs/reproduce_all.log
```

Resume after interrupted compute: `bash scripts/kdd2027/reproduce_all.sh`.
After temporary weights are lost:
`bash scripts/kdd2027/restore_and_reproduce.sh`.
This restores weights without discarding persistent per-image checkpoints.
Do not launch a second copy while the tmux job is active. Do not claim formal
ASR until its full target-evaluation rows are present and the report is COMPLETE.

Co-Attack/SGA original joint text-image retrieval protocols remain incompatible
with an unchanged image-only targeted10-class table. IAU-FOA paper/source is now
identified; its five-proxy family-exclusion adaptation and source integration
remain a subsequent task, not a completed reproduction or failed empirical result.
