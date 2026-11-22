#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/src"
exec /home/ubuntu/exper-tran/.venv-a100/bin/python -u scripts/kdd2027/protocol_smoke10/campaign.py "$@"
