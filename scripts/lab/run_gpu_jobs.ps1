# Start (or resume) the long GPU jobs on the lab PC (ROADMAP §7.3 step 6). Both are resumable, so re-running is safe.
#   powershell -File scripts\lab\run_gpu_jobs.ps1 [-DataRoot D:\wildtriage-data] [-CacheRoot D:\wt-env]
# Runs A-05 (BioCLIP 2 embeddings, dev then test) and A-10 (SAM 3 on 150 SA-FARI test clips) side by side
# (about 9 GB of the 20 GB VRAM). Logs go to <CacheRoot>\tmp\*.log.
param(
    [string]$DataRoot = "D:\wildtriage-data",
    [string]$CacheRoot = "D:\wt-env",
    [string]$HfTokenPath = "$env:USERPROFILE\.cache\huggingface\token"
)
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$env:HF_HOME = "$CacheRoot\hf"; $env:HF_TOKEN_PATH = $HfTokenPath; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TEMP = "$CacheRoot\tmp"; $env:TMP = $env:TEMP; $env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
Set-Location $Root

$embed = {
    param($Root, $Py, $DataRoot, $Log)
    Set-Location $Root
    foreach ($s in "dev", "test") {
        & $Py research/data_prep/extract_embeddings.py --manifest "research/manifests/lila_${s}_sub.parquet" `
            --output-dir "$DataRoot\embeddings\lila_$s" --batch-size 128 --num-workers 24 --device cuda --skip-existing 2>&1 |
            Select-String -Pattern "progress|Saved|ERROR|Traceback|Error|skipping" | ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $Log
    }
}
$exp2 = {
    param($Root, $Py, $DataRoot, $Log)
    Set-Location $Root
    & $Py research/experiments/exp2_video_counts.py run --n-clips 150 --out "$DataRoot\exp2" 2>&1 |
        Select-String -Pattern '^\[\d|Traceback|Error' | ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 $Log
}
Start-Job -Name embeddings -ScriptBlock $embed -ArgumentList $Root, $Py, $DataRoot, "$env:TEMP\embed.log" | Out-Null
Start-Job -Name exp2 -ScriptBlock $exp2 -ArgumentList $Root, $Py, $DataRoot, "$env:TEMP\exp2_run.log" | Out-Null
Write-Host "Started jobs 'embeddings' and 'exp2'. Check with: Get-Job; Get-Content $env:TEMP\exp2_run.log -Tail 3"
Write-Host "Jobs live in this PowerShell session; keep it open (or use Start-Process powershell for a detached window)."
