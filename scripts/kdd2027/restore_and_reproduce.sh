#!/usr/bin/env bash
# Restore the disposable weight cache before resuming a preserved root-disk campaign.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
bash scripts/kdd2027/run_baselines.sh download
bash scripts/kdd2027/run_baselines.sh download-targets
exec bash scripts/kdd2027/reproduce_all.sh
