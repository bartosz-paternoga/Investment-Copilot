#!/usr/bin/env bash
set -euo pipefail

# Restore ONLY into a separate recovery database. Never point this script at the
# production DATABASE_URL; this is intentionally named RESTORE_DATABASE_URL.
: "${RESTORE_DATABASE_URL:?Set RESTORE_DATABASE_URL to a separate, empty recovery database}"
BACKUP_FILE="${1:-}"
if [[ -z "$BACKUP_FILE" || ! -f "$BACKUP_FILE" ]]; then
  echo "Usage: RESTORE_DATABASE_URL='postgresql://.../recovery_db' $0 backups/postgres/file.dump" >&2
  exit 2
fi
command -v pg_restore >/dev/null 2>&1 || { echo "pg_restore is required (install PostgreSQL client tools)." >&2; exit 1; }
pg_restore --list "$BACKUP_FILE" >/dev/null
read -r -p "Type RESTORE to restore into the database specified by RESTORE_DATABASE_URL: " CONFIRM
if [[ "$CONFIRM" != "RESTORE" ]]; then echo "Cancelled."; exit 1; fi
pg_restore --dbname="$RESTORE_DATABASE_URL" --no-owner --no-privileges --clean --if-exists "$BACKUP_FILE"
echo "Restore completed. Validate login, user records, and tenant portfolio data before any cutover."
