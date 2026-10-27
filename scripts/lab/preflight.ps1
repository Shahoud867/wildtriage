# University-PC preflight (ROADMAP §7.2). Prints PASS/FAIL/WARN per check; exits 1 if any required check fails.
#   powershell -ExecutionPolicy Bypass -File scripts\lab\preflight.ps1 [-DataDrive D:] [-MinFreeGB 200] [-DryRun]
# -DryRun skips the GPU and network checks so the script itself can be tested on any machine.
param(
    [string]$DataDrive = "D:",
    [int]$MinFreeGB = 200,
    [switch]$DryRun
)
$ErrorActionPreference = "Continue"
$script:Failed = 0

function Report($Name, $Status, $Detail) {
    Write-Host ("[{0}] {1}: {2}" -f $Status, $Name, $Detail)
    if ($Status -eq "FAIL") { $script:Failed++ }
}

# 1. GPU
if ($DryRun) {
    Report "gpu" "WARN" "skipped (dry run)"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) -join "; "
    if ($gpu -match "4000 Ada") { Report "gpu" "PASS" $gpu } else { Report "gpu" "WARN" "not an RTX 4000 Ada: $gpu" }
} else {
    Report "gpu" "FAIL" "nvidia-smi not found"
}

# 2. Python 3.12 (system, py launcher, or uv-managed)
$py12 = $false
foreach ($cmd in @("py -3.12 --version", "python --version")) {
    try { if ((Invoke-Expression "$cmd 2>&1") -match "3\.12") { $py12 = $true; break } } catch {}
}
if (-not $py12 -and (Get-Command uv -ErrorAction SilentlyContinue)) {
    $py12 = [bool]((uv python list --only-installed 2>$null) -match "3\.12")
}
if ($py12) { Report "python" "PASS" "3.12 available" }
elseif (Get-Command uv -ErrorAction SilentlyContinue) { Report "python" "WARN" "3.12 missing; run: uv python install 3.12" }
else { Report "python" "FAIL" "no Python 3.12 and no uv (pip install uv into a venv, or use the bundle)" }

# 3. Disk space
try {
    $drive = Get-PSDrive -Name $DataDrive.TrimEnd(":") -PSProvider FileSystem -ErrorAction Stop
    $free = [int]($drive.Free / 1GB)
    if ($free -ge $MinFreeGB) { Report "disk" "PASS" "$free GB free on $DataDrive" }
    else { Report "disk" "FAIL" "$free GB free on $DataDrive, need $MinFreeGB" }
} catch {
    Report "disk" "FAIL" "drive $DataDrive not found"
}
$sysFree = [int]((Get-PSDrive -Name C -PSProvider FileSystem).Free / 1GB)
if ($sysFree -lt 5) { Report "system-drive" "WARN" "C: has only $sysFree GB free; point UV_CACHE_DIR, HF_HOME, TEMP at $DataDrive" }

# 4. Network
$hosts = @("https://huggingface.co", "https://storage.googleapis.com", "https://download.pytorch.org", "https://pypi.org")
foreach ($h in $hosts) {
    if ($DryRun) { Report "net $h" "WARN" "skipped (dry run)"; continue }
    try {
        $code = (Invoke-WebRequest $h -UseBasicParsing -TimeoutSec 10 -Method Head).StatusCode
        Report "net $h" "PASS" "HTTP $code"
    } catch {
        $r = $_.Exception.Response
        if ($r) { Report "net $h" "PASS" "HTTP $([int]$r.StatusCode)" } else { Report "net $h" "FAIL" $_.Exception.Message }
    }
}

# 5. Writable folders
foreach ($dir in @((Join-Path "$DataDrive\" "wildtriage-data"), (Join-Path $env:USERPROFILE "wildtriage-data"))) {
    try {
        New-Item -ItemType Directory -Force $dir -ErrorAction Stop | Out-Null
        $probe = Join-Path $dir ".write-test"
        Set-Content -Path $probe -Value "ok" -ErrorAction Stop
        Remove-Item $probe -Force
        Report "writable $dir" "PASS" "ok"
        break
    } catch {
        Report "writable $dir" "WARN" "not writable"
    }
}

# 6. Optional tools
foreach ($tool in @("git", "docker", "node", "uv", "ffmpeg")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) { Report "optional $tool" "PASS" "found" }
    else { Report "optional $tool" "WARN" "not found (optional)" }
}

if ($script:Failed -gt 0) { Write-Host "$($script:Failed) required check(s) FAILED."; exit 1 }
Write-Host "All required checks passed."
