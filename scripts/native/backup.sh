#!/usr/bin/env bash
# Database dump + copy of media to a second disk. Schedule with cron (Linux) or launchd (macOS).
# Override with WT_PG_BIN, WT_PG_PORT, WT_MEDIA_ROOT and WT_BACKUP_ROOT.
set -euo pipefail

PG_BIN="${WT_PG_BIN:-$(dirname "$(command -v pg_dump || echo /usr/bin/pg_dump)")}"
PORT="${WT_PG_PORT:-5432}"
MEDIA_ROOT="${WT_MEDIA_ROOT:-$HOME/wildtriage-data/media}"
BACKUP_ROOT="${WT_BACKUP_ROOT:?set WT_BACKUP_ROOT to a folder on another disk}"
STAMP="$(date +%Y%m%d-%H%M)"

mkdir -p "$BACKUP_ROOT/db" "$BACKUP_ROOT/media"
"$PG_BIN/pg_dump" -h localhost -U wildtriage -p "$PORT" -Fc -f "$BACKUP_ROOT/db/wildtriage-$STAMP.dump" wildtriage
rsync -a --delete "$MEDIA_ROOT/" "$BACKUP_ROOT/media/"
echo "Backup written to $BACKUP_ROOT"
