# Baseline execution

Worktree `/home/ubuntu/exper-tran-kdd2027`, branch
`experiment/kdd2027-baseline-reproduction`. Read CODEX_HANDOFF and the three audit
reports here first. This implementation never edits Ours, the paper, or old results.

Use the existing Python3.11 venv; the only added dependency is
`kmeans-pytorch==0.3`. All public checkpoint revisions are pinned. Checkpoints go to
the large temporary mount; receipts, PNGs, raw target outputs and logs go to root disk.

```bash
bash scripts/kdd2027/run_baselines.sh download
bash scripts/kdd2027/run_baselines.sh download-targets
bash scripts/kdd2027/run_baselines.sh preflight
bash scripts/kdd2027/run_baselines.sh smoke
bash scripts/kdd2027/run_baselines.sh pilot
bash scripts/kdd2027/reproduce_all.sh
```

The final command runs tests, audits, smoke, pilot, matched50/full150,
then M/FOA native300/full150, with evaluation of all frozen images. No commercial
API/CLI calls. Target checkpoint downloads can run CPU-side in parallel using the
explicit `download-targets` command above. No existing workload is killed. One
baseline GPU owner with a72GiB maximum; occupied GPU returns an explicit blocker.

Rerun the identical command to continue at the first missing/invalid image.
SHA256-verified valid PNG receipts are reused; corrupt PNGs are recomputed. Source,
manifest, model, reference and budget changes create a new versioned run directory.
Targets load only after required PNGs are frozen, with explicit precision=bf16 and
recursive dtype/quantization audit. Missing checkpoints or failed cells produce a
nonzero exit and N/A results, never fake completed/zero-ASR rows.

`--methods`, `--steps`, `--split`, `--proxies`, `--cache-dir`, `--output`,
`--ours-root`, `--data-root`, `--upstream-root` are explicit. Changing proxy identities
makes a distinct adaptation; never call a one-proxy smoke the full three-proxy method.
Attack-only commands `attacks`, `smoke`, `pilot` return success for that stage only;
they do not assert target evaluation is complete. `report` makes compact CSV/JSON/MD.
The scripts resume after interruption; no claim of OS-reboot automatic supervision.

Matched50 is **not the original M/FOA300 budget**. Both are queued and separately
identified. Final target results are not consulted to tune losses or select images.
Pilot uses historical screen10 only for engineering validation; it is not an
independent test. All unsupported methods fail closed. Co-Attack/SGA compatibility
and IAU-FOA verification limitations remain explicit in the compatibility report.
