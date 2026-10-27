#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Run ./start.sh once first." >&2
  exit 1
fi
exec .venv/bin/python scripts/market_data_worker.py "$@"
