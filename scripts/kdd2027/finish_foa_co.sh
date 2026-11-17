#!/usr/bin/env bash
# Narrow user-authorized queue: ImageNet Co-Attack and corrected FOA only.
set -uo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PRIMARY_ML_CKA_STRICT_BF16=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON_BIN="${BASELINE_PYTHON:-/home/ubuntu/exper-tran/.venv-a100/bin/python}"
OUT=/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction
LOG="$OUT/extension/finish_foa_co"
mkdir -p "$LOG"
exec 9>>"$OUT/reproduction_owner.lock"
flock -n 9 || { echo 'Another campaign owns admission'; exit 1; }
failed=0
for method in co_attack foa_attack; do
  method_failed=0
  for phase in attack evaluate; do
    echo "$(date -u +%FT%TZ) START $method $phase"
    "$PYTHON_BIN" -u scripts/kdd2027/extension/worker.py --method "$method" --dataset imagenet --phase "$phase" >>"$LOG/${method}_${phase}.log" 2>&1
    code=$?
    echo "$(date -u +%FT%TZ) EXIT $method $phase $code"
    if (( code != 0 )); then failed=1; method_failed=1; break; fi
  done
  "$PYTHON_BIN" - "$method" <<'PY'
import sys
sys.path.insert(0,'scripts/kdd2027/extension')
from task_protocol import condition
from worker import lock_job,OUT
from metrics import summarize
config=condition(sys.argv[1],'imagenet','default',False)
config['pilot']=False
run,_,_=lock_job(config)
summarize(run,OUT)
PY
  if (( $? != 0 )); then failed=1; fi
  echo "$(date -u +%FT%TZ) METHOD_FINISHED $method failure=$method_failed"
done
bash scripts/kdd2027/run_7x3x3.sh status
exit "$failed"
