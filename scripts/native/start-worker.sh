#!/usr/bin/env bash
# Starts one worker process. Run one per queue set:
#   bash scripts/native/start-worker.sh cpu              # any machine
#   bash scripts/native/start-worker.sh gpu              # MacBook (WT_DEVICE=mps) or the T1000 PC
#   bash scripts/native/start-worker.sh gpu,gpu-heavy    # Linux machine with a large NVIDIA GPU
# On the Mac the gpu-heavy queue is refused: SAM 3 and DepthSAM need CUDA (lab PC, or molab).
set -euo pipefail

QUEUES="${1:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ "$(uname -s)" == Darwin ]]; then
  if [[ ",$QUEUES," == *",gpu-heavy,"* ]]; then
    echo "The gpu-heavy queue needs CUDA (SAM 3, DepthSAM). Run it on the lab PC or molab; use 'gpu' here." >&2
    exit 2
  fi
  # Some PyTorch ops are not implemented on MPS yet; run those on the CPU instead of failing.
  export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

cd "$ROOT/backend"
exec "$PY" -m app.workers.runner --queues "$QUEUES"
