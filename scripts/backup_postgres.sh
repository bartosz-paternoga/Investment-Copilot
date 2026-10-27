#!/usr/bin/env bash
set -euo pipefail

# Run from a trusted machine/runner. The dump contains all user portfolios and
# authentication hashes; keep it encrypted and outside Git/repository artifacts.
: "${DATABASE_URL:?Set DATABASE_URL to the managed PostgreSQL connection string}"
OUT_DIR="${PG_BACKUP_DIR:-backups/postgres}"
mkdir -p "$OUT_DIR"
chmod 700 "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_DIR/investment-pilot-$STAMP.dump"

command -v pg_dump >/dev/null 2>&1 || { echo "pg_dump is required (install PostgreSQL client tools)." >&2; exit 1; }
umask 077
pg_dump "$DATABASE_URL" --format=custom --no-owner --no-privileges --file="$OUT"
pg_restore --list "$OUT" >/dev/null
sha256sum "$OUT" > "$OUT.sha256"
chmod 600 "$OUT" "$OUT.sha256"
echo "Backup created and archive listing verified: $OUT"
echo "Store this dump in encrypted, access-controlled storage separate from this machine."
