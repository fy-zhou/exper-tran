#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${BASELINE_PYTHON:-/home/ubuntu/exper-tran/.venv-a100/bin/python}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PRIMARY_ML_CKA_STRICT_BF16=1
exec "$PYTHON_BIN" -m primary_ml_cka.baselines.campaign "$@"
