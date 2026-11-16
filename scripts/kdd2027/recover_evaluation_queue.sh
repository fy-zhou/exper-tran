#!/usr/bin/env bash
# Operational recovery of the original immutable runs; no attack source migration.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PRIMARY_ML_CKA_STRICT_BF16=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${BASELINE_PYTHON:-/home/ubuntu/exper-tran/.venv-a100/bin/python}"
OUT_DIR=/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction
exec 9>"$OUT_DIR/reproduction_owner.lock"
flock -n 9 || exit 2
exec > >(tee -a "$OUT_DIR/logs/evaluation_recovery.log") 2>&1
"$PYTHON_BIN" scripts/kdd2027/evaluate_frozen_run.py --run "$OUT_DIR/runs/final150_50_206015cb1202ed50f1"
"$PYTHON_BIN" scripts/kdd2027/evaluate_frozen_run.py --run "$OUT_DIR/runs/final150_300_87488fbb38aae77f36"
native_status=0
bash scripts/kdd2027/run_baselines.sh run-all --steps 300 --methods m_attack foa_attack --output "$OUT_DIR" || native_status=$?
"$PYTHON_BIN" scripts/kdd2027/evaluate_frozen_run.py --run "$OUT_DIR/runs/final150_300_87488fbb38aae77f36"
exit "$native_status"
