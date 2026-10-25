# Applies migrations, then serves the API on http://localhost:8000 (docs at /docs).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location (Join-Path $Root "backend")
& $Py -m alembic upgrade head
& $Py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
