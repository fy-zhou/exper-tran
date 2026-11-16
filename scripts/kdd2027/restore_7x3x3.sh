#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${BASELINE_PYTHON:-/home/ubuntu/exper-tran/.venv-a100/bin/python}"
"$PYTHON_BIN" scripts/kdd2027/prepare_upstreams.py
if [[ ! -d /home/ubuntu/baseline-upstreams/kmeans_pinned ]]; then
  git clone --no-checkout https://github.com/subhadarship/kmeans_pytorch.git /home/ubuntu/baseline-upstreams/kmeans_pinned
  git -C /home/ubuntu/baseline-upstreams/kmeans_pinned checkout --detach f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805
fi
bash scripts/kdd2027/run_baselines.sh download
bash scripts/kdd2027/run_baselines.sh download-targets
"$PYTHON_BIN" - <<'PY'
import sys
sys.path.insert(0,'scripts/kdd2027/extension')
from native_methods import EXTRAS
from huggingface_hub import snapshot_download
for spec in EXTRAS.values():
 snapshot_download(spec['repo_id'],revision=spec['revision'],cache_dir='/ephemeral/kdd2027/hf-cache/hub',allow_patterns=['*.json','*.txt','*.model','*.safetensors'],max_workers=2)
PY
"$PYTHON_BIN" scripts/kdd2027/extension/prepare_datasets.py
exec bash scripts/kdd2027/run_7x3x3.sh start --minimum-hours 8
