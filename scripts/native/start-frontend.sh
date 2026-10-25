#!/usr/bin/env bash
# Dev server on http://localhost:5173 (proxies /api to :8000).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm ci --no-audit --no-fund
fi
exec npm run dev
