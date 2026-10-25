# Portable PostgreSQL without admin rights.
# 1. Download the Windows "zip archive" binaries of PostgreSQL 16 (EDB "PostgreSQL binaries" page) and unzip,
#    e.g. to D:\tools\pgsql. 2. Run this script. The first run initialises the cluster and asks for a password;
#    put the same password in .env (WT_DATABASE_URL=postgresql+psycopg://wildtriage:<password>@localhost:5432/wildtriage).
param(
    [string]$PgBin = "D:\tools\pgsql\bin",
    [string]$DataDir = "D:\wildtriage-data\pg",
    [int]$Port = 5432
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $DataDir "PG_VERSION"))) {
    New-Item -ItemType Directory -Force $DataDir | Out-Null
    & (Join-Path $PgBin "initdb.exe") -D $DataDir -U wildtriage --auth=scram-sha-256 --pwprompt -E UTF8
}
& (Join-Path $PgBin "pg_ctl.exe") -D $DataDir -l (Join-Path $DataDir "server.log") -o "-p $Port" start
Start-Sleep -Seconds 2
& (Join-Path $PgBin "createdb.exe") -U wildtriage -p $Port wildtriage
if ($LASTEXITCODE -ne 0) { Write-Host "createdb did not succeed. That is fine if 'wildtriage' already exists; otherwise check the password." }
Write-Host "PostgreSQL running on port $Port. Stop it with: pg_ctl -D $DataDir stop"
