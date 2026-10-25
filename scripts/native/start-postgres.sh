#!/usr/bin/env bash
# Local PostgreSQL without admin rights.
#   macOS: `brew install postgresql@16` (Homebrew installs under your user) or Postgres.app.
#   Linux: your distribution's postgresql package, or the EDB binaries unpacked anywhere.
# The first run initialises a cluster and asks for a password; put the same password in .env
# (WT_DATABASE_URL=postgresql+psycopg://wildtriage:<password>@localhost:5432/wildtriage).
# Override paths with WT_PG_BIN, WT_PG_DATA and WT_PG_PORT.
set -euo pipefail

DATA_DIR="${WT_PG_DATA:-$HOME/wildtriage-data/pg}"
PORT="${WT_PG_PORT:-5432}"
PG_BIN="${WT_PG_BIN:-}"

if [[ -z "$PG_BIN" ]]; then
  if command -v pg_ctl >/dev/null 2>&1; then
    PG_BIN="$(dirname "$(command -v pg_ctl)")"
  else
    for candidate in /opt/homebrew/opt/postgresql@16/bin /usr/local/opt/postgresql@16/bin \
      /Applications/Postgres.app/Contents/Versions/latest/bin /usr/lib/postgresql/16/bin; do
      if [[ -x "$candidate/pg_ctl" ]]; then PG_BIN="$candidate"; break; fi
    done
  fi
fi
if [[ -z "$PG_BIN" || ! -x "$PG_BIN/pg_ctl" ]]; then
  echo "pg_ctl not found. Install PostgreSQL 16 or set WT_PG_BIN to its bin folder." >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/PG_VERSION" ]]; then
  mkdir -p "$DATA_DIR"
  "$PG_BIN/initdb" -D "$DATA_DIR" -U wildtriage --auth=scram-sha-256 --pwprompt -E UTF8
fi
if "$PG_BIN/pg_ctl" -D "$DATA_DIR" status >/dev/null 2>&1; then
  echo "PostgreSQL is already running for $DATA_DIR."
else
  "$PG_BIN/pg_ctl" -D "$DATA_DIR" -l "$DATA_DIR/server.log" -o "-p $PORT" start
  sleep 2
fi
if ! "$PG_BIN/createdb" -h localhost -U wildtriage -p "$PORT" wildtriage; then
  echo "createdb did not succeed. That is fine if 'wildtriage' already exists; otherwise check the password."
fi
echo "PostgreSQL running on port $PORT. Stop it with: $PG_BIN/pg_ctl -D $DATA_DIR stop"
