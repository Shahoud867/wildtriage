# Dev server on http://localhost:5173 (proxies /api to :8000).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location (Join-Path $Root "frontend")
if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund }
npm run dev
