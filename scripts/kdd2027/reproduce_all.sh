#!/usr/bin/env bash
# Complete ordered campaign. No paper edits, no commercial calls, no hidden method substitution.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${BASELINE_PYTHON:-/home/ubuntu/exper-tran/.venv-a100/bin/python}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
OUT_DIR="${BASELINE_OUTPUT_ROOT:-/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction}"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/audits"
exec 9>"$OUT_DIR/reproduction_owner.lock"
flock -n 9 || { echo 'Another reproduction owner is active'; exit 2; }
exec > >(tee -a "$OUT_DIR/logs/reproduce_all.log") 2>&1
cd "$REPO_DIR"
"$PYTHON_BIN" -m pytest -q tests/unit/test_kdd_baseline_iterative.py tests/unit/test_kdd_baseline_protocol.py tests/unit/test_kdd_baseline_recovery.py
"$PYTHON_BIN" -m ruff check src/primary_ml_cka/baselines scripts/kdd2027 tests/unit/test_kdd_baseline*py
"$PYTHON_BIN" scripts/kdd2027/prepare_upstreams.py
"$PYTHON_BIN" scripts/kdd2027/audit_ours.py --output "$OUT_DIR/audits/ours.json"
# Public pinned proxy weights only; target downloads are explicit in the campaign command below.
bash scripts/kdd2027/run_baselines.sh download --output "$OUT_DIR"
bash scripts/kdd2027/run_baselines.sh smoke --output "$OUT_DIR"
bash scripts/kdd2027/run_baselines.sh pilot --output "$OUT_DIR"
status=0
# Missing target checkpoints are documented; attacks can still be frozen and resumed.
bash scripts/kdd2027/run_baselines.sh run-all --output "$OUT_DIR" || status=$?
RUN50_DIR="$(bash scripts/kdd2027/run_baselines.sh prepare --output "$OUT_DIR")"
"$PYTHON_BIN" scripts/kdd2027/evaluate_frozen_run.py --run "$RUN50_DIR" --output "$OUT_DIR" || status=$?
# Original 300-step settings are additional, explicitly separated from 50-step controls.
bash scripts/kdd2027/run_baselines.sh run-all --steps 300 --methods m_attack foa_attack --output "$OUT_DIR" || status=$?
RUN300_DIR="$(bash scripts/kdd2027/run_baselines.sh prepare --steps 300 --methods m_attack foa_attack --output "$OUT_DIR")"
"$PYTHON_BIN" scripts/kdd2027/evaluate_frozen_run.py --run "$RUN300_DIR" --output "$OUT_DIR" || status=$?
exit "$status"
