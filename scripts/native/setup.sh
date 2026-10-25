#!/usr/bin/env bash
# One-time setup on macOS or Linux, without admin rights.
#   bash scripts/native/setup.sh mac    # MacBook (Apple silicon): PyTorch with MPS, real models
#   bash scripts/native/setup.sh cuda   # Linux machine with an NVIDIA GPU: CUDA 12.8 wheels
#   bash scripts/native/setup.sh cpu    # any machine: stub models only
# Needs uv (https://docs.astral.sh/uv/, installs per-user) and, for the frontend, Node.js 22+.
set -euo pipefail

MACHINE="${1:-cpu}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

case "$MACHINE" in
  cpu | mac | cuda) ;;
  *) echo "usage: setup.sh [cpu|mac|cuda]" >&2; exit 2 ;;
esac
if [[ "$MACHINE" == mac && ( "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ) ]]; then
  echo "'mac' expects an Apple silicon Mac (Darwin arm64); use 'cpu' or 'cuda' here." >&2
  exit 2
fi

uv venv --python 3.12 .venv
case "$MACHINE" in
  cpu)
    uv pip install --python "$PY" -e "ml[dev]" -e "backend[dev]" pillow pandas
    ;;
  mac)
    # The macOS wheels on PyPI include the MPS backend; no extra index needed.
    uv pip install --python "$PY" torch torchvision
    uv pip install --python "$PY" -e "ml[gpu,dev]" -e "backend[dev]" pandas
    ;;
  cuda)
    uv pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python "$PY" -e "ml[gpu,dev]" -e "backend[dev]" pandas
    ;;
esac

if [[ "$MACHINE" != cpu ]]; then
  "$PY" -c 'import torch
from wildtriage_ml.device import pick_device, pick_dtype
d = pick_device()
print("torch", torch.__version__, "| device:", d, "| dtype:", pick_dtype(d))'
fi

if [[ ! -f "$ROOT/.env" ]]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "Created .env from .env.example. Edit it for this machine:"
  case "$MACHINE" in
    mac) echo "  WT_MODEL_BACKEND=real  WT_DEVICE=mps  WT_GPU_VRAM_BUDGET_GB=12" ;;
    cuda) echo "  WT_MODEL_BACKEND=real  WT_DEVICE=cuda  WT_GPU_VRAM_BUDGET_GB=<GPU memory minus ~2>" ;;
    cpu) echo "  WT_MODEL_BACKEND=stub" ;;
  esac
fi
echo "Done. Next: scripts/native/start-postgres.sh, then start-api.sh, start-worker.sh and start-frontend.sh."
