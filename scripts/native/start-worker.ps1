# Starts one worker process. Run one per queue set:
#   start-worker.ps1 -Queues cpu                 # any machine
#   start-worker.ps1 -Queues gpu                 # T1000 (WT_GPU_VRAM_BUDGET_GB=3.5 in .env)
#   start-worker.ps1 -Queues "gpu,gpu-heavy"     # lab PC (WT_GPU_VRAM_BUDGET_GB=18)
param([string]$Queues = "cpu")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
Set-Location (Join-Path $Root "backend")
& $Py -m app.workers.runner --queues $Queues
