#!/usr/bin/env python3
"""Scheduled market-data ingestion worker for Investment Pilot.

Run persistently (Docker Compose starts it as a separate service) or with
--once from cron. It shares the application's SQLite database and never trades.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/market_data_worker.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from portfolio import (
    build_portfolio, get_history, get_quotes, initialize_db, list_transactions,
    list_watchlist, log_market_data_run, save_snapshot, transaction_types_summary,
)


def configured_universe(transactions: pd.DataFrame) -> list[str]:
    symbols: set[str] = set()
    if not transactions.empty:
        summary = transaction_types_summary(transactions, "PLN")
        symbols.update(t for t, p in summary.get("positions", {}).items() if p.get("quantity", 0) > 1e-10)
    watch = list_watchlist()
    if not watch.empty:
        symbols.update(watch["ticker"].astype(str).str.strip().str.upper())
    extra = os.getenv("ETF_MARKET_DATA_EXTRA_TICKERS", "^GSPC,^STOXX50E,^IXIC")
    symbols.update(s.strip().upper() for s in extra.split(",") if s.strip())
    return sorted(s for s in symbols if s)


def run_refresh() -> dict[str, object]:
    initialize_db()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start_mark = time.monotonic()
    status = "ok"
    errors: list[str] = []
    history_rows = 0
    transactions = list_transactions()
    symbols = configured_universe(transactions)
    try:
        quotes = get_quotes(symbols, force_refresh=True, max_age_seconds=0)
        if not quotes.empty:
            failed = quotes.loc[quotes["status"].isin(["Unavailable", "Incomplete", "Stale"]), "ticker"].astype(str).tolist()
            if failed:
                errors.append("Quote unavailable/incomplete/stale: " + ", ".join(failed))
        period = os.getenv("ETF_MARKET_DATA_HISTORY_PERIOD", "5y").strip() or "5y"
        history = get_history(symbols, period=period, force_refresh=True)
        history_rows = int(history.notna().sum().sum()) if not history.empty else 0
        if history.empty:
            errors.append("No historical prices returned")
        else:
            missing_history = sorted(set(symbols) - set(map(str, history.columns)))
            if missing_history:
                errors.append("Historical prices missing: " + ", ".join(missing_history))
            stale_history = []
            for symbol in symbols:
                if symbol not in history.columns or history[symbol].dropna().empty:
                    continue
                last_observation = pd.Timestamp(history[symbol].dropna().index.max()).normalize()
                if (pd.Timestamp.now().normalize() - last_observation).days > 5:
                    stale_history.append(symbol)
            if stale_history:
                errors.append("Historical prices are stale: " + ", ".join(stale_history))
        if not transactions.empty:
            portfolio = build_portfolio(transactions, quotes, "PLN")
            if not portfolio.empty and bool(portfolio["market_value_base"].notna().all()):
                save_snapshot(portfolio, "PLN", "Scheduled market-data refresh")
        if errors:
            status = "partial"
    except Exception as exc:  # Worker must record a failed run, then remain alive.
        status = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    detail = "; ".join(errors) if errors else f"Refreshed {len(symbols)} symbols in {time.monotonic()-start_mark:.1f}s"
    log_market_data_run(started, finished, status, len(symbols), history_rows, detail)
    result = {"status": status, "tickers": len(symbols), "history_rows": history_rows, "duration_seconds": round(time.monotonic()-start_mark, 1), "detail": detail}
    print(f"[{finished}] {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Investment Pilot scheduled market-data ingestion")
    parser.add_argument("--once", action="store_true", help="Run once and exit (useful with cron/systemd timer)")
    parser.add_argument("--interval-seconds", type=int, default=int(os.getenv("ETF_MARKET_DATA_REFRESH_SECONDS", "86400")),
                        help="Delay between scheduled runs; default 86400 seconds")
    args = parser.parse_args()
    interval = max(300, args.interval_seconds)
    if args.once:
        run_refresh()
        return
    print(f"Market-data worker started; refresh interval={interval}s", flush=True)
    while True:
        run_refresh()
        time.sleep(interval)


if __name__ == "__main__":
    main()
