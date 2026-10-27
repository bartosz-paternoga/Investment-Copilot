#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ ! -f .env ]; then cp .env.example .env; echo "Created .env. Add OPENROUTER_API_KEY to enable AI analysis."; fi
# This launcher is for local single-user development only. Hosted deployments must configure DATABASE_URL.
if [ -z "${DATABASE_URL:-}" ]; then export ETF_APP_LOCAL_DEV_MODE=1; fi
streamlit run app.py
