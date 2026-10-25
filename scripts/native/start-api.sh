#!/usr/bin/env bash
# Applies migrations, then serves the API on http://localhost:8000 (docs at /docs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT/backend"
"$PY" -m alembic upgrade head
exec "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
