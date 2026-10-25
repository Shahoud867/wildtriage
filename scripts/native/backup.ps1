# Nightly backup: database dump + copy of media to a second disk. Schedule with Task Scheduler (user-level task).
param(
    [string]$PgBin = "D:\tools\pgsql\bin",
    [string]$MediaRoot = "D:\wildtriage-data\media",
    [string]$BackupRoot = "E:\wildtriage-backup",
    [int]$Port = 5432
)
$ErrorActionPreference = "Stop"
$Stamp = Get-Date -Format "yyyyMMdd-HHmm"
New-Item -ItemType Directory -Force (Join-Path $BackupRoot "db") | Out-Null
& (Join-Path $PgBin "pg_dump.exe") -U wildtriage -p $Port -Fc -f (Join-Path $BackupRoot "db\wildtriage-$Stamp.dump") wildtriage
robocopy $MediaRoot (Join-Path $BackupRoot "media") /MIR /NFL /NDL /NJH /NJS | Out-Null
Write-Host "Backup written to $BackupRoot"
