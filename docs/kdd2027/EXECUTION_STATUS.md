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

## 2026-11-16 15:02 UTC — evaluation-priority recovery

50-step attacks: PGD150/150, MI150/150, M150/150, FOA147/150.
FOA failed on three images because a Lloyd iteration produced an empty cluster;
the method failed explicitly rather than dropping a local loss or changing K.
The old launcher incorrectly allowed these failures to block evaluation of ALL
completed methods. No formal ASR was available at that point. This is an
orchestration defect, not evidence of zero transfer.

300-step M has also completed150/150. Its images are preserved. Only the owned
native attack worker was interrupted (FOA300 had1 frozen image) to prioritize
frozen-image evaluation. An unfinished image is recomputed from its original seed.
No attack sources, scientific locks, existing PNGs or target protocol changed.

Recovery: `scripts/kdd2027/evaluate_frozen_run.py` verifies the original source and
cohort hashes, then groups independently completed methods. A partially failed
method may be evaluated only once every original image has either a valid frozen
PNG or an explicit failed-attempt record. Its full planned N remains150, missing
outcomes remain missing, and its aggregate ASR is N/A until complete. The subset
is artifact-based, never prediction-based; the evaluation plan is frozen before
model access. Three new regression tests cover method independence, pending
images and source-lock rejection;36 tests pass in total.

Actual tmux queue now: evaluate50 (three complete methods, then FOA147);
evaluate300 M150; resume remaining FOA300 attacks; evaluate native completed or
explicitly partial methods. Logs: `logs/evaluation_recovery.log`.
The standard reproduction launcher also calls this recovery path after each
budget, so one failed method cannot suppress other methods' reports again.

## 2026-11-17 — broad queue paused; narrow continuation authorized

Historical RUNNING text above is superseded. Broad63-cell campaign paused at02:07UTC by user; report and full structured evidence: [pause snapshot](results/paused_2026-11-17/REPORT.md). Nine current formal cells complete. A separate user instruction authorized ImageNet Co-Attack and revised FOA full150 completion; detached `kdd2027-foa-co` runs only those two. Other formal tasks remain paused. No commercial evaluation was launched.
