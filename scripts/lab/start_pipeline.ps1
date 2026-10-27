# Start the unattended GPU pipeline fully detached from this terminal (survives closing it; not a logoff or reboot).
#   powershell -File scripts\lab\start_pipeline.ps1
# Re-running is safe: finished steps are skipped, and it refuses to start a second supervisor.
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "lab\\pipeline.py" }
if ($running) { Write-Host "Supervisor already running (pid $($running.ProcessId))."; exit 0 }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Log = "D:\wt-env\tmp\pipeline_supervisor.log"
$cmd = "cmd /c `"`"$Py`" `"$Root\scripts\lab\pipeline.py`" >> `"$Log`" 2>&1`""
# Win32_Process.Create starts the process outside this shell's job object, so it is not killed with it.
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Root }
Write-Host "Started supervisor (pid $($r.ProcessId)). Status: D:\wildtriage-data\pipeline_status.json; report: D:\wildtriage-data\PIPELINE_REPORT.md"
