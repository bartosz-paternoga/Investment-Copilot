#!/usr/bin/env bash
set -euo pipefail

# Run from the project root. Uses SQLite's online backup API so a live WAL-mode
# database is copied consistently. Output is stored outside the app data folder.
DB_PATH="${ETF_APP_DB_PATH:-${ETF_APP_DATA_DIR:-data}/portfolio.sqlite3}"
OUT_DIR="${ETF_BACKUP_DIR:-backups}"
mkdir -p "$OUT_DIR"
if [[ ! -f "$DB_PATH" ]]; then
  echo "Database not found: $DB_PATH" >&2
  exit 1
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/portfolio-$STAMP.sqlite3"
python - "$DB_PATH" "$OUT" <<'PY'
import sqlite3, sys
source, destination = sys.argv[1:]
with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
    src.backup(dst)
    result = dst.execute('PRAGMA integrity_check').fetchone()[0]
    if result != 'ok':
        raise SystemExit(f'Backup integrity check failed: {result}')
print(f'Backup created and verified: {destination}')
PY
