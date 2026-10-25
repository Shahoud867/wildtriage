# One-time setup without admin rights (works on the managed lab PC).
#   powershell -ExecutionPolicy Bypass -File scripts\native\setup.ps1 -Machine lab     # RTX 4000 Ada
#   powershell -ExecutionPolicy Bypass -File scripts\native\setup.ps1 -Machine t1000   # Quadro T1000
#   powershell -ExecutionPolicy Bypass -File scripts\native\setup.ps1 -Machine laptop  # no CUDA, stub models
# Needs uv (https://docs.astral.sh/uv/, installs per-user) and, for the frontend, Node.js 22+.
param([ValidateSet("laptop", "t1000", "lab")][string]$Machine = "laptop")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

uv venv --python 3.12 .venv
if ($Machine -eq "laptop") {
    uv pip install --python $Py -e "ml[dev]" -e "backend[dev]" pillow pandas
} else {
    uv pip install --python $Py torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python $Py -e "ml[gpu,dev]" -e "backend[dev]" pandas
    & $Py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok); print('GPU:', torch.cuda.get_device_name(0) if ok else '-'); print('bf16:', ok and torch.cuda.is_bf16_supported()); print('archs:', torch.cuda.get_arch_list())"
}

if (-not (Test-Path (Join-Path $Root ".env"))) {
    Copy-Item (Join-Path $Root ".env.example") (Join-Path $Root ".env")
    Write-Host "Created .env from .env.example. Edit it for this machine (WT_MODEL_BACKEND, WT_GPU_VRAM_BUDGET_GB, WT_DATABASE_URL)."
}
Write-Host "Done. Next: scripts\native\start-postgres.ps1, then start-api.ps1, start-worker.ps1 and start-frontend.ps1."
