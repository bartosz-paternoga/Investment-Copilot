from __future__ import annotations

import os
import sqlite3
import time
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except ImportError:  # Allows accounting tests to run without market-data dependencies.
    yf = None
from dotenv import load_dotenv
from database import connect_database, database_url, active_tenant_schema

load_dotenv()
DATA_DIR = Path(os.getenv("ETF_APP_DATA_DIR", "data"))
DB_PATH = Path(os.getenv("ETF_APP_DB_PATH", str(DATA_DIR / "portfolio.sqlite3")))
LEGACY_HOLDINGS_PATH = DATA_DIR / "holdings.csv"
VALID_TYPES = {"BUY", "SELL", "OPENING", "DIVIDEND", "FEE", "CASH_IN", "CASH_OUT"}
FX_CACHE: dict[tuple[str, str, str], tuple[float, str]] = {}
FX_CACHE_TIME: dict[tuple[str, str, str], float] = {}
FX_CACHE_FETCHED_AT: dict[tuple[str, str, str], str] = {}
_DB_INIT_LOCK = threading.RLock()
_INITIALIZED_DB_KEYS: set[tuple[str, str]] = set()


def _connect():
    """Open the active tenant's database (PostgreSQL in hosted mode)."""
    return connect_database(DB_PATH)


def initialize_db() -> None:
    """Create this user's schema once per process without re-running DDL on each UI rerun."""
    dsn = database_url()
    tenant = active_tenant_schema() if dsn else str(DB_PATH.resolve())
    if dsn and tenant is None:
        raise RuntimeError("No authenticated tenant is active; refusing to initialize PostgreSQL without tenant isolation.")
    key = (dsn or "sqlite", str(tenant))
    with _DB_INIT_LOCK:
        if key in _INITIALIZED_DB_KEYS:
            return
        _initialize_schema_tables()
        if not dsn:
            _migrate_legacy_holdings_once()
        _INITIALIZED_DB_KEYS.add(key)


def _initialize_schema_tables() -> None:
    with _connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL DEFAULT '',
            transaction_type TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 0,
            unit_price REAL NOT NULL DEFAULT 0,
            currency TEXT NOT NULL DEFAULT 'PLN',
            amount REAL NOT NULL DEFAULT 0,
            fee REAL NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK (transaction_type IN ('BUY','SELL','OPENING','DIVIDEND','FEE','CASH_IN','CASH_OUT'))
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            as_of TEXT NOT NULL,
            base_currency TEXT NOT NULL,
            market_value REAL,
            cash_value REAL,
            account_value REAL,
            quote_count INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT ''
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(trade_date, id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(ticker)")
        con.execute("""CREATE TABLE IF NOT EXISTS instruments (
            ticker TEXT PRIMARY KEY, display_name TEXT NOT NULL DEFAULT '', isin TEXT NOT NULL DEFAULT '',
            exchange TEXT NOT NULL DEFAULT '', asset_class TEXT NOT NULL DEFAULT 'ETF',
            notes TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY, added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            note TEXT NOT NULL DEFAULT '', sort_order INTEGER NOT NULL DEFAULT 0
        )""")
        # Existing installs predate user-controlled watchlist ordering.
        if database_url():
            watchlist_columns = {str(row["column_name"]) for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='watchlist'"
            ).fetchall()}
        else:
            watchlist_columns = {str(row["name"]) for row in con.execute("PRAGMA table_info(watchlist)").fetchall()}
        if "sort_order" not in watchlist_columns:
            con.execute("ALTER TABLE watchlist ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
            existing_watchlist = con.execute("SELECT ticker FROM watchlist ORDER BY added_at DESC, ticker").fetchall()
            for order, row in enumerate(existing_watchlist, start=1):
                con.execute("UPDATE watchlist SET sort_order=? WHERE ticker=?", (order, row["ticker"]))
        con.execute("""CREATE TABLE IF NOT EXISTS fx_rates (
            source_currency TEXT NOT NULL, target_currency TEXT NOT NULL, requested_date TEXT NOT NULL,
            rate REAL NOT NULL, returned_date TEXT NOT NULL, cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(source_currency,target_currency,requested_date))""")
        con.execute("""CREATE TABLE IF NOT EXISTS quote_cache (
            ticker TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', currency TEXT NOT NULL DEFAULT '',
            price REAL, previous_close REAL, day_change_pct REAL, price_date TEXT NOT NULL DEFAULT '',
            fetched_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'Yahoo Finance via yfinance',
            status TEXT NOT NULL DEFAULT 'Unavailable', error TEXT NOT NULL DEFAULT ''
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS market_prices (
            ticker TEXT NOT NULL, price_date TEXT NOT NULL, adjusted_close REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'Yahoo Finance via yfinance', fetched_at TEXT NOT NULL,
            PRIMARY KEY(ticker,price_date)
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices(price_date)")
        con.execute("""CREATE TABLE IF NOT EXISTS fx_history (
            source_currency TEXT NOT NULL, target_currency TEXT NOT NULL, rate_date TEXT NOT NULL,
            rate REAL NOT NULL, returned_date TEXT NOT NULL, cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(source_currency,target_currency,rate_date)
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS etf_holdings (
            fund_ticker TEXT NOT NULL, constituent_ticker TEXT NOT NULL, constituent_name TEXT NOT NULL DEFAULT '',
            weight_pct REAL NOT NULL, source_name TEXT NOT NULL, source_url TEXT NOT NULL DEFAULT '',
            as_of TEXT NOT NULL, imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(fund_ticker,constituent_ticker,as_of)
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS market_data_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
            status TEXT NOT NULL, refreshed_tickers INTEGER NOT NULL DEFAULT 0,
            history_rows INTEGER NOT NULL DEFAULT 0, detail TEXT NOT NULL DEFAULT ''
        )""")

def _migrate_legacy_holdings_once() -> None:
    """Preserve the old CSV holdings if the new transaction ledger is empty."""
    if not LEGACY_HOLDINGS_PATH.exists():
        return
    with _connect() as con:
        count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        if count:
            return
        try:
            old = pd.read_csv(LEGACY_HOLDINGS_PATH)
        except Exception:
            return
        if old.empty or "ticker" not in old.columns or "quantity" not in old.columns:
            return
        today = date.today().isoformat()
        for _, row in old.iterrows():
            ticker = str(row.get("ticker", "") or "").strip().upper()
            qty = _num(row.get("quantity"), 0)
            basis = _num(row.get("cost_basis"), 0)
            currency = str(row.get("cost_currency", "PLN") or "PLN").upper()
            if ticker and qty > 0:
                con.execute("""INSERT INTO transactions
                    (trade_date,ticker,transaction_type,quantity,unit_price,currency,notes)
                    VALUES (?,?,?,?,?,?,?)""",
                    (today, ticker, "OPENING", qty, max(basis, 0), currency,
                     "Migrated from the previous holdings.csv. Exact original purchase dates/FX were not available."))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def save_instrument(ticker: str, display_name: str = "", isin: str = "", exchange: str = "", asset_class: str = "ETF", notes: str = "") -> None:
    initialize_db()
    ticker = str(ticker or "").strip().upper()
    if not ticker or len(ticker) > 40:
        raise ValueError("Enter a valid ticker or listing symbol (maximum 40 characters).")
    with _connect() as con:
        con.execute("""INSERT INTO instruments(ticker,display_name,isin,exchange,asset_class,notes)
            VALUES(?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET
            display_name=excluded.display_name, isin=excluded.isin, exchange=excluded.exchange,
            asset_class=excluded.asset_class, notes=excluded.notes, updated_at=CURRENT_TIMESTAMP""",
            (ticker, str(display_name or "")[:200], str(isin or "")[:32], str(exchange or "")[:80],
             str(asset_class or "ETF")[:40], str(notes or "")[:2000]))


def list_instruments() -> pd.DataFrame:
    initialize_db()
    with _connect() as con:
        rows = con.execute("SELECT * FROM instruments ORDER BY ticker").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["ticker", "display_name", "isin", "exchange", "asset_class", "notes", "updated_at"])


def save_etf_holdings(
    fund_ticker: str, holdings: pd.DataFrame, *, source_name: str, source_url: str = "", as_of: str
) -> int:
    """Persist one sourced constituent snapshot, replacing that fund/date atomically.

    Expected columns: constituent_ticker, weight_pct; optional constituent_name.
    Weights are percentage points (e.g. 7.25 means 7.25%), not decimals.
    """
    initialize_db()
    fund = str(fund_ticker or "").strip().upper()
    source_name = str(source_name or "").strip()[:200]
    source_url = str(source_url or "").strip()[:1000]
    as_of = str(as_of or "")[:10]
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError("As-of date must be a valid ISO date.") from exc
    if not fund or len(fund) > 40:
        raise ValueError("Fund ticker is required (maximum 40 characters).")
    if not source_name:
        raise ValueError("Enter the holdings data source, such as the issuer factsheet name.")
    if source_url and not source_url.lower().startswith(("https://", "http://")):
        raise ValueError("Source URL must start with http:// or https://.")
    if holdings is None or holdings.empty or "constituent_ticker" not in holdings.columns or "weight_pct" not in holdings.columns:
        raise ValueError("CSV must contain constituent_ticker and weight_pct columns.")
    prepared = holdings.copy()
    prepared["constituent_ticker"] = prepared["constituent_ticker"].fillna("").astype(str).str.strip().str.upper()
    prepared["weight_pct"] = pd.to_numeric(prepared["weight_pct"], errors="coerce")
    if "constituent_name" not in prepared.columns:
        prepared["constituent_name"] = ""
    prepared["constituent_name"] = prepared["constituent_name"].fillna("").astype(str).str.strip().str.slice(0, 200)
    if (prepared["constituent_ticker"].eq("").any() or prepared["constituent_ticker"].str.len().gt(80).any()
            or prepared["weight_pct"].isna().any() or (~np.isfinite(prepared["weight_pct"])).any()
            or (prepared["weight_pct"] < 0).any() or (prepared["weight_pct"] > 100).any()):
        raise ValueError("Every constituent needs a ticker and a numeric weight between 0 and 100.")
    if prepared["constituent_ticker"].duplicated().any():
        raise ValueError("Constituent tickers must be unique within one fund snapshot.")
    total = float(prepared["weight_pct"].sum())
    if total <= 0:
        raise ValueError("The sum of constituent weights must be greater than zero.")
    rows = [(fund, row.constituent_ticker, row.constituent_name, float(row.weight_pct), source_name, source_url, as_of)
            for row in prepared.itertuples(index=False)]
    with _connect() as con:
        con.execute("DELETE FROM etf_holdings WHERE fund_ticker=? AND as_of=?", (fund, as_of))
        con.executemany("""INSERT INTO etf_holdings
            (fund_ticker,constituent_ticker,constituent_name,weight_pct,source_name,source_url,as_of)
            VALUES(?,?,?,?,?,?,?)""", rows)
    return len(rows)


def list_etf_holdings(latest_only: bool = True) -> pd.DataFrame:
    """Read imported ETF constituents with source/date provenance."""
    initialize_db()
    with _connect() as con:
        if latest_only:
            rows = con.execute("""SELECT h.* FROM etf_holdings h
                JOIN (SELECT fund_ticker, MAX(as_of) AS latest_as_of FROM etf_holdings GROUP BY fund_ticker) x
                ON h.fund_ticker=x.fund_ticker AND h.as_of=x.latest_as_of
                ORDER BY h.fund_ticker,h.weight_pct DESC""").fetchall()
        else:
            rows = con.execute("SELECT * FROM etf_holdings ORDER BY fund_ticker,as_of DESC,weight_pct DESC").fetchall()
    columns = ["fund_ticker", "constituent_ticker", "constituent_name", "weight_pct", "source_name", "source_url", "as_of", "imported_at"]
    return pd.DataFrame([dict(r) for r in rows], columns=columns)


def log_market_data_run(started_at: str, finished_at: str, status: str, refreshed_tickers: int,
                        history_rows: int, detail: str = "") -> None:
    initialize_db()
    with _connect() as con:
        con.execute("""INSERT INTO market_data_runs(started_at,finished_at,status,refreshed_tickers,history_rows,detail)
            VALUES(?,?,?,?,?,?)""", (started_at, finished_at, status, int(refreshed_tickers), int(history_rows), str(detail or "")[:4000]))


def read_market_data_runs(limit: int = 50) -> pd.DataFrame:
    initialize_db()
    with _connect() as con:
        rows = con.execute("SELECT * FROM market_data_runs ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "started_at", "finished_at", "status", "refreshed_tickers", "history_rows", "detail"])


def market_data_cache_summary() -> dict[str, Any]:
    initialize_db()
    with _connect() as con:
        quotes = con.execute("SELECT COUNT(*) AS n, MAX(fetched_at) AS latest FROM quote_cache").fetchone()
        prices = con.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT ticker) AS tickers, MIN(price_date) AS first_date, MAX(price_date) AS last_date FROM market_prices").fetchone()
        fx = con.execute("SELECT COUNT(*) AS n, MAX(cached_at) AS latest FROM fx_history").fetchone()
        constituents = con.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT fund_ticker) AS funds FROM etf_holdings").fetchone()
    return {"quote_rows": int(quotes["n"] or 0), "last_quote_fetch": quotes["latest"],
            "history_rows": int(prices["n"] or 0), "history_tickers": int(prices["tickers"] or 0),
            "history_first_date": prices["first_date"], "history_last_date": prices["last_date"],
            "fx_fixings": int(fx["n"] or 0), "last_fx_cache_update": fx["latest"],
            "constituent_rows": int(constituents["n"] or 0), "constituent_funds": int(constituents["funds"] or 0)}


def add_watchlist_ticker(ticker: str, note: str = "") -> None:
    initialize_db()
    ticker = str(ticker or "").strip().upper()
    if not ticker or len(ticker) > 40:
        raise ValueError("Enter a valid ticker or listing symbol.")
    with _connect() as con:
        row = con.execute("SELECT sort_order FROM watchlist WHERE ticker=?", (ticker,)).fetchone()
        if row is None:
            next_order = con.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM watchlist").fetchone()[0]
            con.execute("INSERT INTO watchlist(ticker,note,sort_order) VALUES(?,?,?)",
                        (ticker, str(note or "")[:500], int(next_order)))
        else:
            con.execute("UPDATE watchlist SET note=? WHERE ticker=?", (str(note or "")[:500], ticker))


def remove_watchlist_ticker(ticker: str) -> None:
    initialize_db()
    with _connect() as con:
        con.execute("DELETE FROM watchlist WHERE ticker=?", (str(ticker).strip().upper(),))


def list_watchlist() -> pd.DataFrame:
    initialize_db()
    with _connect() as con:
        rows = con.execute("SELECT * FROM watchlist ORDER BY sort_order, added_at DESC, ticker").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["ticker", "added_at", "note", "sort_order"])


def save_watchlist_order(tickers: Iterable[str]) -> None:
    """Save the complete user-selected order for the current watchlist."""
    symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Each instrument must have a different order number.")
    with _connect() as con:
        current = {str(row["ticker"]).upper() for row in con.execute("SELECT ticker FROM watchlist").fetchall()}
        if set(symbols) != current:
            raise ValueError("The watchlist changed while you were editing it. Refresh the page and try again.")
        for order, symbol in enumerate(symbols, start=1):
            con.execute("UPDATE watchlist SET sort_order=? WHERE ticker=?", (order, symbol))


def list_transactions() -> pd.DataFrame:
    initialize_db()
    with _connect() as con:
        rows = con.execute("SELECT * FROM transactions ORDER BY trade_date DESC, id DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=[
        "id", "trade_date", "ticker", "transaction_type", "quantity", "unit_price", "currency", "amount", "fee", "notes", "created_at"
    ])


def resolve_position_currency(
    ticker: str,
    quotes: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    fallback: str = "PLN",
) -> str:
    """Resolve a currency for manually priced trades without guessing EUR/PLN.

    A valid market-data listing currency is the best choice for a manual trade
    price. When quote metadata is missing, fall back to the most recent ledger
    currency for the ticker, then to the caller-provided reporting currency.
    """
    symbol = str(ticker or "").strip().upper()

    def valid_currency(value: Any) -> str | None:
        currency = str(value or "").strip().upper()
        return currency if len(currency) == 3 and currency.isalpha() else None

    if symbol and isinstance(quotes, pd.DataFrame) and not quotes.empty and {"ticker", "currency"}.issubset(quotes.columns):
        quote_rows = quotes.loc[quotes["ticker"].astype(str).str.strip().str.upper() == symbol]
        if not quote_rows.empty:
            for value in reversed(quote_rows["currency"].tolist()):
                currency = valid_currency(value)
                if currency:
                    return currency

    if symbol and isinstance(transactions, pd.DataFrame) and not transactions.empty and {"ticker", "currency"}.issubset(transactions.columns):
        ledger_rows = transactions.loc[transactions["ticker"].astype(str).str.strip().str.upper() == symbol].copy()
        sort_columns = [column for column in ("trade_date", "id") if column in ledger_rows.columns]
        if sort_columns:
            ledger_rows = ledger_rows.sort_values(sort_columns, kind="stable")
        for value in reversed(ledger_rows["currency"].tolist()):
            currency = valid_currency(value)
            if currency:
                return currency

    return valid_currency(fallback) or "PLN"


def add_transaction(tx: dict[str, Any]) -> int:
    initialize_db()
    kind = str(tx.get("transaction_type", "")).upper().strip()
    ticker = str(tx.get("ticker", "") or "").upper().strip()
    tx_date = str(tx.get("trade_date", date.today().isoformat()))[:10]
    currency = str(tx.get("currency", "PLN") or "PLN").upper().strip()
    qty = _num(tx.get("quantity"))
    price = _num(tx.get("unit_price"))
    amount = _num(tx.get("amount"))
    fee = _num(tx.get("fee"))
    notes = str(tx.get("notes", "") or "")[:2000]

    if kind not in VALID_TYPES:
        raise ValueError("Unsupported transaction type.")
    try:
        date.fromisoformat(tx_date)
    except ValueError as exc:
        raise ValueError("Transaction date must be a valid date.") from exc
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError("Use a three-letter ISO currency code, e.g. PLN or EUR.")
    if fee < 0 or qty < 0 or price < 0 or amount < 0:
        raise ValueError("Quantity, price, amount and fee cannot be negative.")
    if kind in {"BUY", "SELL", "OPENING"}:
        if not ticker or qty <= 0:
            raise ValueError(f"{kind} needs a ticker and a positive quantity.")
        if kind in {"BUY", "SELL"} and price <= 0:
            raise ValueError(f"{kind} needs a positive unit price.")
        if kind == "SELL" and fee > qty * price:
            raise ValueError("Sell fees cannot exceed sale proceeds.")
    elif kind in {"DIVIDEND", "FEE", "CASH_IN", "CASH_OUT"}:
        if amount <= 0:
            raise ValueError(f"{kind} needs a positive amount.")
        if kind in {"DIVIDEND", "FEE"} and ticker and not ticker.replace(".", "").isalnum():
            raise ValueError("Ticker contains unsupported characters.")

    with _connect() as con:
        cur = con.execute("""INSERT INTO transactions
            (trade_date,ticker,transaction_type,quantity,unit_price,currency,amount,fee,notes)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (tx_date, ticker, kind, qty, price, currency, amount, fee, notes))
        new_id = int(cur.lastrowid)
        # Prevent accidental short positions, including when inserting a back-dated sale.
        running: dict[str, float] = {}
        rows = con.execute("SELECT trade_date,ticker,transaction_type,quantity,id FROM transactions ORDER BY trade_date,id").fetchall()
        for row in rows:
            symbol = str(row["ticker"] or "").upper()
            if not symbol or row["transaction_type"] not in {"BUY", "SELL", "OPENING"}:
                continue
            running[symbol] = running.get(symbol, 0.0) + (float(row["quantity"]) if row["transaction_type"] != "SELL" else -float(row["quantity"]))
            if running[symbol] < -1e-8:
                con.execute("DELETE FROM transactions WHERE id=?", (new_id,))
                raise ValueError(f"This sale would make {symbol} holdings negative on {row['trade_date']}. Add the missing earlier purchase or correct the sale.")
    return new_id


def replace_quick_position(ticker: str, quantity: float, average_cost: float, currency: str, trade_date: str, display_name: str = "") -> str:
    """Create/update/remove a simple Quick Holdings position without duplicating it.

    Positions with BUY/SELL or other transaction types are intentionally protected:
    their quantities must be changed through the transaction ledger instead.
    A quantity of zero removes the opening-position entries for this ticker.
    """
    initialize_db()
    symbol = str(ticker or "").strip().upper()
    qty, price = _num(quantity, -1), _num(average_cost, -1)
    ccy = str(currency or "").strip().upper()
    try:
        date.fromisoformat(str(trade_date)[:10])
    except ValueError as exc:
        raise ValueError("Position date must be a valid date.") from exc
    if not symbol or len(symbol) > 40:
        raise ValueError("Enter a valid ticker or listing symbol.")
    if qty < 0 or price < 0:
        raise ValueError("Quantity and average cost cannot be negative.")
    if len(ccy) != 3 or not ccy.isalpha():
        raise ValueError("Use a three-letter currency code.")
    with _connect() as con:
        rows = con.execute("SELECT id,transaction_type FROM transactions WHERE UPPER(ticker)=? ORDER BY id", (symbol,)).fetchall()
        if any(row["transaction_type"] != "OPENING" for row in rows):
            raise ValueError(f"{symbol} has BUY/SELL or other ledger activity. Use Transactions to change it safely.")
        con.execute("DELETE FROM transactions WHERE UPPER(ticker)=?", (symbol,))
        if qty > 0:
            con.execute("""INSERT INTO transactions(trade_date,ticker,transaction_type,quantity,unit_price,currency,amount,fee,notes)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (str(trade_date)[:10], symbol, "OPENING", qty, price, ccy, 0.0, 0.0, "Quick holdings position (editable snapshot)"))
    if display_name.strip():
        save_instrument(symbol, display_name=display_name.strip())
    return "removed" if qty == 0 else ("updated" if rows else "added")


def remove_ticker_from_portfolio(ticker: str) -> int:
    """Remove all ledger rows for a ticker; callers must obtain explicit confirmation."""
    initialize_db()
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        raise ValueError("Select a ticker to remove.")
    with _connect() as con:
        count = int(con.execute("SELECT COUNT(*) FROM transactions WHERE UPPER(ticker)=?", (symbol,)).fetchone()[0])
        if not count:
            return 0
        con.execute("DELETE FROM transactions WHERE UPPER(ticker)=?", (symbol,))
    return count


def delete_transaction(transaction_id: int) -> None:
    initialize_db()
    with _connect() as con:
        row = con.execute("SELECT ticker FROM transactions WHERE id=?", (int(transaction_id),)).fetchone()
        if row is None:
            raise ValueError("Transaction was not found.")
        con.execute("DELETE FROM transactions WHERE id=?", (int(transaction_id),))
        running: dict[str, float] = {}
        rows = con.execute("SELECT trade_date,ticker,transaction_type,quantity FROM transactions ORDER BY trade_date,id").fetchall()
        for event in rows:
            symbol = str(event["ticker"] or "").upper()
            if not symbol or event["transaction_type"] not in {"BUY", "SELL", "OPENING"}:
                continue
            running[symbol] = running.get(symbol, 0.0) + (float(event["quantity"]) if event["transaction_type"] != "SELL" else -float(event["quantity"]))
            if running[symbol] < -1e-8:
                raise ValueError(f"Deleting this transaction would make {symbol} holdings negative. The deletion was not applied.")


def transaction_types_summary(transactions: pd.DataFrame, base_currency: str = "PLN") -> dict[str, Any]:
    realized = 0.0
    realized_pl_known = True
    dividends = 0.0
    fees_total = 0.0
    contributions = 0.0
    withdrawals = 0.0
    missing_fx: list[str] = []
    qty_state: dict[str, float] = {}
    basis_state: dict[str, float] = {}
    basis_known: dict[str, bool] = {}
    cash: dict[str, float] = {}
    if transactions.empty:
        return {"realized_pl": 0.0, "dividends": 0.0, "fees": 0.0, "contributions": 0.0,
                "withdrawals": 0.0, "cash": {}, "missing_fx": [], "positions": {}}
    # Batch historical FX by currency/date range rather than making one HTTP call
    # per ledger row. This materially improves large-ledger app reloads.
    tx_dates = pd.to_datetime(transactions.get("trade_date", pd.Series(dtype=str)), errors="coerce").dropna()
    fx_histories: dict[str, pd.Series] = {}
    if not tx_dates.empty:
        first_tx_date, last_tx_date = tx_dates.min().date().isoformat(), tx_dates.max().date().isoformat()
        currencies = set(transactions.get("currency", pd.Series([base_currency])).fillna(base_currency).astype(str).str.upper())
        for ccy in currencies:
            if ccy != str(base_currency).upper():
                try:
                    fx_histories[ccy] = get_fx_history(ccy, base_currency, first_tx_date, last_tx_date)
                except Exception:
                    fx_histories[ccy] = pd.Series(dtype=float)

    def transaction_fx(ccy: str, tx_date: str) -> float | None:
        if ccy == str(base_currency).upper():
            return 1.0
        series = fx_histories.get(ccy, pd.Series(dtype=float))
        if not series.empty:
            try:
                series = series.sort_index()
                target_date = pd.Timestamp(tx_date)
                position = int(series.index.searchsorted(target_date, side="right")) - 1
                if position >= 0:
                    fixing_date = pd.Timestamp(series.index[position])
                    rate = float(series.iloc[position])
                    if (target_date - fixing_date).days <= 7 and np.isfinite(rate) and rate > 0:
                        return rate
            except (TypeError, ValueError, IndexError):
                pass
        # Fallback for unsupported currencies or unusually long reference-rate gaps.
        return fx_rate(ccy, base_currency, tx_date)

    for _, tx in transactions.sort_values(["trade_date", "id"]).iterrows():
        kind = str(tx["transaction_type"]).upper()
        ticker = str(tx.get("ticker", "") or "").upper()
        ccy = str(tx.get("currency", base_currency) or base_currency).upper()
        tx_date = str(tx.get("trade_date", date.today().isoformat()))[:10]
        rate = transaction_fx(ccy, tx_date)
        if rate is None:
            missing_fx.append(f"{tx_date} {ccy}/{base_currency} transaction #{int(tx['id'])}")
        qty, price, amount, fee = map(_num, [tx.get("quantity"), tx.get("unit_price"), tx.get("amount"), tx.get("fee")])
        # Cash movements are tracked in each currency; OPENING records add units without cash movement.
        if kind in {"BUY", "SELL"}:
            cash_delta = (-(qty * price + fee) if kind == "BUY" else qty * price - fee)
            cash[ccy] = cash.get(ccy, 0.0) + cash_delta
            if rate is not None:
                fees_total += fee * rate
        elif kind == "DIVIDEND":
            cash[ccy] = cash.get(ccy, 0.0) + amount
            if rate is not None:
                dividends += amount * rate
        elif kind == "FEE":
            cash[ccy] = cash.get(ccy, 0.0) - amount
            if rate is not None:
                fees_total += amount * rate
        elif kind == "CASH_IN":
            cash[ccy] = cash.get(ccy, 0.0) + amount
            if rate is not None:
                contributions += amount * rate
        elif kind == "CASH_OUT":
            cash[ccy] = cash.get(ccy, 0.0) - amount
            if rate is not None:
                withdrawals += amount * rate

        if kind not in {"BUY", "SELL", "OPENING"} or not ticker:
            continue
        old_qty = qty_state.get(ticker, 0.0)
        old_basis = basis_state.get(ticker, 0.0)
        known = basis_known.get(ticker, True)
        if kind in {"BUY", "OPENING"}:
            new_cost = (qty * price + fee) * rate if rate is not None else float("nan")
            qty_state[ticker] = old_qty + qty
            if kind == "OPENING" and price <= 0:
                basis_known[ticker] = False
                basis_state[ticker] = old_basis
            elif rate is None:
                basis_known[ticker] = False
                basis_state[ticker] = old_basis
            else:
                basis_state[ticker] = old_basis + new_cost
                basis_known[ticker] = known
        elif kind == "SELL":
            if old_qty > 0 and known and rate is not None:
                cost_sold = old_basis * min(qty, old_qty) / old_qty
                proceeds = (qty * price - fee) * rate
                realized += proceeds - cost_sold
                basis_state[ticker] = old_basis - cost_sold
            else:
                realized_pl_known = False
                basis_known[ticker] = False
            qty_state[ticker] = old_qty - qty
    positions: dict[str, dict[str, Any]] = {}
    for ticker, qty in qty_state.items():
        basis = basis_state.get(ticker, 0.0)
        known = basis_known.get(ticker, True)
        positions[ticker] = {"quantity": max(qty, 0.0), "cost_basis_base": basis if known else None,
                             "average_cost_base": basis / qty if known and qty > 0 else None,
                             "cost_basis_known": bool(known)}
    return {"realized_pl": realized, "realized_pl_known": realized_pl_known, "dividends": dividends, "fees": fees_total,
            "contributions": contributions, "withdrawals": withdrawals, "cash": cash,
            "missing_fx": sorted(set(missing_fx)), "positions": positions}


def position_quantities() -> dict[str, float]:
    tx = list_transactions()
    if tx.empty:
        return {}
    positions = transaction_types_summary(tx, "PLN")["positions"]
    return {t: p["quantity"] for t, p in positions.items()}


def fx_rate(source: str, target: str, on_date: str | None = None, *, force_refresh: bool = False) -> float | None:
    source = str(source or "").upper().strip()
    target = str(target or "").upper().strip()
    if not source or not target:
        return None
    if source == target:
        return 1.0
    req_date = (str(on_date)[:10] if on_date else "latest")
    key = (source, target, req_date)
    if key in FX_CACHE:
        age = time.time() - FX_CACHE_TIME.get(key, 0.0)
        if not force_refresh and (req_date != "latest" or age < 1800):
            return FX_CACHE[key][0]
        FX_CACHE.pop(key, None)
        FX_CACHE_TIME.pop(key, None)
        FX_CACHE_FETCHED_AT.pop(key, None)
    try:
        cached = None
        if not force_refresh:
            with _connect() as con:
                cached = con.execute("""SELECT rate,returned_date,cached_at FROM fx_rates
                    WHERE source_currency=? AND target_currency=? AND requested_date=?
                    AND (? != 'latest' OR cached_at >= datetime('now','-30 minutes'))""", (*key, req_date)).fetchone()
        if cached is not None:
            FX_CACHE[key] = (float(cached["rate"]), str(cached["returned_date"]))
            FX_CACHE_TIME[key] = time.time()
            cached_at = cached["cached_at"]
            FX_CACHE_FETCHED_AT[key] = cached_at.isoformat() if hasattr(cached_at, "isoformat") else str(cached_at)
            return float(cached["rate"])
    except Exception:
        # Database tables may not yet exist when fx_rate is called independently.
        pass
    date_part = "latest" if req_date == "latest" else req_date
    try:
        response = requests.get(f"https://api.frankfurter.dev/v1/{date_part}", params={"base": source, "symbols": target}, timeout=12)
        response.raise_for_status()
        payload = response.json()
        rate = float(payload["rates"][target])
        returned_date = str(payload.get("date", req_date))
        if not np.isfinite(rate) or rate <= 0:
            return None
        FX_CACHE[key] = (rate, returned_date)
        FX_CACHE_TIME[key] = time.time()
        FX_CACHE_FETCHED_AT[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save_fx_rate(source, target, req_date, rate, returned_date)
        return rate
    except Exception:
        try:
            response = requests.get(f"https://api.frankfurter.dev/v1/{date_part}", params={"base": target, "symbols": source}, timeout=12)
            response.raise_for_status()
            payload = response.json()
            inverse = float(payload["rates"][source])
            rate = 1.0 / inverse
            returned_date = str(payload.get("date", req_date))
            if not np.isfinite(rate) or rate <= 0:
                return None
            FX_CACHE[key] = (rate, returned_date)
            FX_CACHE_TIME[key] = time.time()
            FX_CACHE_FETCHED_AT[key] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _save_fx_rate(source, target, req_date, rate, returned_date)
            return rate
        except Exception:
            return None


def _save_fx_rate(source: str, target: str, requested_date: str, rate: float, returned_date: str) -> None:
    try:
        with _connect() as con:
            con.execute("""INSERT INTO fx_rates(source_currency,target_currency,requested_date,rate,returned_date)
                VALUES(?,?,?,?,?) ON CONFLICT(source_currency,target_currency,requested_date) DO UPDATE SET
                rate=excluded.rate, returned_date=excluded.returned_date, cached_at=CURRENT_TIMESTAMP""",
                (source, target, requested_date, float(rate), returned_date))
    except Exception:
        # A missing/unavailable cache table should not prevent live valuations.
        pass


def fx_returned_date(source: str, target: str, on_date: str | None = None) -> str | None:
    if source == target:
        return on_date or date.today().isoformat()
    fx_rate(source, target, on_date)
    pair = (str(source).upper(), str(target).upper(), str(on_date)[:10] if on_date else "latest")
    cached = FX_CACHE.get(pair)
    if cached:
        return cached[1]
    inverse = FX_CACHE.get((pair[1], pair[0], pair[2]))
    return inverse[1] if inverse else None


def fx_rate_info(
    source: str, target: str, on_date: str | None = None, *, force_refresh: bool = False
) -> dict[str, Any] | None:
    """Return an FX rate with its provider fixing date and cache/fetch timestamp."""
    source = str(source or "").upper().strip()
    target = str(target or "").upper().strip()
    if not source or not target:
        return None
    rate = fx_rate(source, target, on_date, force_refresh=force_refresh)
    if rate is None:
        return None
    req_date = str(on_date)[:10] if on_date else "latest"
    if source == target:
        return {"rate": float(rate), "returned_date": req_date if req_date != "latest" else date.today().isoformat(),
                "fetched_at": None}
    key = (source, target, req_date)
    fetched_at = FX_CACHE_FETCHED_AT.get(key)
    returned_date = fx_returned_date(source, target, on_date)
    if not fetched_at:
        # Covers rates loaded by another code path or retained in a database cache.
        try:
            with _connect() as con:
                cached = con.execute("""SELECT returned_date,cached_at FROM fx_rates
                    WHERE source_currency=? AND target_currency=? AND requested_date=?""", key).fetchone()
            if cached is not None:
                returned_date = str(cached["returned_date"])
                cached_at = cached["cached_at"]
                fetched_at = cached_at.isoformat() if hasattr(cached_at, "isoformat") else str(cached_at)
                FX_CACHE_FETCHED_AT[key] = fetched_at
        except Exception:
            pass
    return {"rate": float(rate), "returned_date": returned_date, "fetched_at": fetched_at}


def fx_spot_pair_tickers(currencies: Iterable[str], target: str, *, inverse: bool = False) -> list[str]:
    """Return direct or inverse Yahoo Finance pair symbols for current spot quotes."""
    target = str(target or "").upper().strip()
    pairs: list[str] = []
    for raw_currency in currencies:
        source = str(raw_currency or "").upper().strip()
        if not source or not target or source == target:
            continue
        pairs.append(f"{target}{source}=X" if inverse else f"{source}{target}=X")
    return list(dict.fromkeys(pairs))


def fx_spot_rate_info(source: str, target: str, quotes: pd.DataFrame | None) -> dict[str, Any] | None:
    """Resolve a current Yahoo FX quote, returning quote metadata and inverse support.

    The pair quote is used for current portfolio values and daily P/L. Historical
    performance calculations continue to use dated daily reference FX fixings.
    """
    source = str(source or "").upper().strip()
    target = str(target or "").upper().strip()
    if not source or not target:
        return None
    if source == target:
        return {"rate": 1.0, "previous_rate": 1.0, "returned_date": "—", "fetched_at": None,
                "ticker": "", "kind": "same currency"}
    if quotes is None or quotes.empty or "ticker" not in quotes:
        return None
    indexed = quotes.drop_duplicates("ticker", keep="last").set_index("ticker", drop=False)
    for ticker, inverse in ((f"{source}{target}=X", False), (f"{target}{source}=X", True)):
        if ticker not in indexed.index:
            continue
        row = indexed.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        try:
            price = float(row.get("price"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        try:
            previous = float(row.get("previous_close"))
        except (TypeError, ValueError):
            previous = np.nan
        if inverse:
            price = 1.0 / price
            previous = 1.0 / previous if np.isfinite(previous) and previous > 0 else np.nan
        return {"rate": price, "previous_rate": previous if np.isfinite(previous) and previous > 0 else None,
                "returned_date": str(row.get("price_date", "") or ""),
                "fetched_at": row.get("fetched_at"), "ticker": ticker, "kind": "spot"}
    return None


def get_fx_history(
    source: str, target: str, start_date: str, end_date: str | None = None, *, force_refresh: bool = False
) -> pd.Series:
    """Return cached/downloaded daily reference FX rates for a currency pair.

    Dates returned by Frankfurter can be earlier than a requested exchange date
    because the reference series only has fixing days. The caller controls any
    short forward-fill policy used when aligning the rate to market prices.
    """
    source = str(source or "").upper().strip()
    target = str(target or "").upper().strip()
    start = str(start_date)[:10]
    end = str(end_date or date.today().isoformat())[:10]
    try:
        date.fromisoformat(start)
        date.fromisoformat(end)
    except ValueError:
        return pd.Series(dtype=float, name=f"{source}_{target}")
    if not source or not target or start > end:
        return pd.Series(dtype=float, name=f"{source}_{target}")
    date_index = pd.date_range(start, end, freq="D")
    if source == target:
        return pd.Series(1.0, index=date_index, name=f"{source}_{target}")
    initialize_db()
    with _connect() as con:
        rows = con.execute("""SELECT rate_date,rate FROM fx_history WHERE source_currency=? AND target_currency=?
            AND rate_date>=? AND rate_date<=? ORDER BY rate_date""", (source, target, start, end)).fetchall()
    cached = {str(r["rate_date"]): float(r["rate"]) for r in rows}
    # Re-fetch when forced or when the cached series does not reach the current
    # end. Historical points are retained and updated idempotently.
    cached_needs_refresh = (not cached) or (pd.Timestamp(end) - pd.Timestamp(max(cached))).days > 5
    cached_needs_start = bool(cached) and pd.Timestamp(min(cached)) > pd.Timestamp(start) + pd.Timedelta(days=5)
    if force_refresh or cached_needs_refresh or cached_needs_start:
        fetched: dict[str, float] = {}
        for base, quote, invert in ((source, target, False), (target, source, True)):
            try:
                response = requests.get(f"https://api.frankfurter.dev/v1/{start}..{end}",
                                        params={"base": base, "symbols": quote}, timeout=20)
                response.raise_for_status()
                payload = response.json()
                for rate_date, rates in (payload.get("rates", {}) or {}).items():
                    value = float(rates[quote])
                    if invert:
                        value = 1.0 / value
                    if np.isfinite(value) and value > 0:
                        fetched[str(rate_date)[:10]] = value
                if fetched:
                    break
            except Exception:
                continue
        if fetched:
            with _connect() as con:
                con.executemany("""INSERT INTO fx_history(source_currency,target_currency,rate_date,rate,returned_date)
                    VALUES(?,?,?,?,?) ON CONFLICT(source_currency,target_currency,rate_date) DO UPDATE SET
                    rate=excluded.rate,returned_date=excluded.returned_date,cached_at=CURRENT_TIMESTAMP""",
                    [(source, target, d, r, d) for d, r in fetched.items()])
            cached.update(fetched)
    if not cached:
        return pd.Series(dtype=float, name=f"{source}_{target}")
    series = pd.Series(cached, dtype=float)
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
    return series.rename(f"{source}_{target}")


def _fetch_quotes_uncached(tickers: Iterable[str] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(tickers, pd.DataFrame):
        symbols = tickers.get("ticker", pd.Series(dtype=str)).astype(str).tolist()
    else:
        symbols = list(tickers)
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in symbols if str(t).strip()))
    cols = ["ticker", "name", "currency", "price", "previous_close", "day_change_pct", "price_date", "fetched_at", "source", "status", "error"]
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if yf is None:
        return pd.DataFrame([{ "ticker": symbol, "name": symbol, "currency": "", "price": np.nan,
            "previous_close": np.nan, "day_change_pct": np.nan, "price_date": "", "fetched_at": fetched_at,
            "source": "Yahoo Finance via yfinance", "status": "Unavailable", "error": "Install requirements.txt (missing yfinance)."}
            for symbol in symbols], columns=cols)
    for ticker in symbols:
        row: dict[str, Any] = {"ticker": ticker, "name": ticker, "currency": "", "price": np.nan,
            "previous_close": np.nan, "day_change_pct": np.nan, "price_date": "", "fetched_at": fetched_at,
            "source": "Yahoo Finance via yfinance", "status": "Unavailable", "error": ""}
        try:
            instrument = yf.Ticker(ticker)
            fast = instrument.fast_info
            currency = fast.get("currency") or ""
            price = fast.get("lastPrice") or fast.get("last_price")
            previous = fast.get("previousClose") or fast.get("previous_close")
            price_date = ""
            history = pd.DataFrame()
            try:
                history = instrument.history(period="5d", auto_adjust=False, actions=False)
            except Exception:
                history = pd.DataFrame()
            if not history.empty and "Close" in history:
                close = history["Close"].dropna()
                if len(close):
                    if price is None or not np.isfinite(_num(price, np.nan)):
                        price = float(close.iloc[-1])
                    if previous is None and len(close) > 1:
                        previous = float(close.iloc[-2])
                    idx = close.index[-1]
                    price_date = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
            name = ticker
            try:
                info = instrument.info
                currency = info.get("currency") or currency
                name = info.get("shortName") or info.get("longName") or name
                previous = previous or info.get("previousClose")
            except Exception:
                pass
            price_val = _num(price, np.nan)
            prev_val = _num(previous, np.nan)
            status = "OK" if np.isfinite(price_val) and price_val > 0 and currency else "Incomplete"
            error = "" if status == "OK" else "No valid price returned or listing currency unavailable"
            if status == "OK" and price_date:
                try:
                    age_days = (date.today() - date.fromisoformat(price_date)).days
                    if age_days > 7:
                        status = "Stale"
                        error = f"Latest daily history is {age_days} calendar days old; confirm before relying on this value."
                except ValueError:
                    pass
            row.update({"name": str(name), "currency": str(currency or "").upper(), "price": price_val,
                        "previous_close": prev_val, "price_date": price_date,
                        "day_change_pct": (price_val / prev_val - 1) * 100 if np.isfinite(price_val) and np.isfinite(prev_val) and prev_val else np.nan,
                        "status": status, "error": error})
        except Exception as exc:
            row["error"] = str(exc)[:250]
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)



QUOTE_COLUMNS = ["ticker", "name", "currency", "price", "previous_close", "day_change_pct", "price_date", "fetched_at", "source", "status", "error"]


def get_quotes(
    tickers: Iterable[str] | pd.DataFrame, *, force_refresh: bool = False,
    max_age_seconds: int = 21600, cached_only: bool = False,
) -> pd.DataFrame:
    """Return quotes from a persistent TTL cache; refresh misses/expired symbols.

    `force_refresh=True` is used by the explicit UI refresh and scheduled worker.
    `cached_only=True` returns any saved quote immediately without provider calls.
    The fetch timestamp is not represented as an exchange/live quote timestamp.
    """
    if isinstance(tickers, pd.DataFrame):
        raw_symbols = tickers.get("ticker", pd.Series(dtype=str)).astype(str).tolist()
    else:
        raw_symbols = list(tickers)
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in raw_symbols if str(t).strip()))
    if not symbols:
        return pd.DataFrame(columns=QUOTE_COLUMNS)
    initialize_db()
    now = datetime.now(timezone.utc)
    cached: dict[str, dict[str, Any]] = {}
    placeholders = ",".join("?" for _ in symbols)
    with _connect() as con:
        rows = con.execute(f"SELECT * FROM quote_cache WHERE ticker IN ({placeholders})", symbols).fetchall()
    for row in rows:
        d = dict(row)
        try:
            stamp = datetime.fromisoformat(str(d.get("fetched_at", "")).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age = (now - stamp).total_seconds()
        except Exception:
            age = float("inf")
        ttl = int(max_age_seconds) if str(d.get("status", "")) != "Unavailable" else min(300, int(max_age_seconds))
        if cached_only or (not force_refresh and age <= max(0, ttl)):
            cached[str(d["ticker"]).upper()] = d
    to_fetch = [] if cached_only else [symbol for symbol in symbols if symbol not in cached]
    fresh_rows: dict[str, dict[str, Any]] = {}
    if to_fetch:
        fetched = _fetch_quotes_uncached(to_fetch)
        for _, row in fetched.iterrows():
            d = row.to_dict()
            ticker = str(d.get("ticker", "")).upper()
            # A provider outage should not destroy a still-usable older quote.
            previous = next((dict(r) for r in rows if str(dict(r).get("ticker", "")).upper() == ticker), None)
            if str(d.get("status", "")) == "Unavailable" and previous and pd.notna(previous.get("price")):
                d = previous
                d["status"] = "Stale"
                d["error"] = (str(d.get("error", "")) + "; refresh failed, showing last cached value")[:250]
            fresh_rows[ticker] = d
        with _connect() as con:
            for d in fresh_rows.values():
                con.execute("""INSERT INTO quote_cache
                    (ticker,name,currency,price,previous_close,day_change_pct,price_date,fetched_at,source,status,error)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET
                    name=excluded.name,currency=excluded.currency,price=excluded.price,
                    previous_close=excluded.previous_close,day_change_pct=excluded.day_change_pct,
                    price_date=excluded.price_date,fetched_at=excluded.fetched_at,source=excluded.source,
                    status=excluded.status,error=excluded.error""",
                    (str(d.get("ticker", "")).upper(), str(d.get("name", "") or ""), str(d.get("currency", "") or ""),
                     _num(d.get("price"), np.nan), _num(d.get("previous_close"), np.nan), _num(d.get("day_change_pct"), np.nan),
                     str(d.get("price_date", "") or ""), str(d.get("fetched_at", now.isoformat(timespec="seconds"))),
                     str(d.get("source", "Yahoo Finance via yfinance") or "Yahoo Finance via yfinance"),
                     str(d.get("status", "Unavailable") or "Unavailable"), str(d.get("error", "") or "")[:250]))
    combined = {**cached, **fresh_rows}
    output = [combined[symbol] for symbol in symbols if symbol in combined]
    result = pd.DataFrame(output, columns=QUOTE_COLUMNS)
    for col in ["price", "previous_close", "day_change_pct"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result

def build_portfolio(
    transactions: pd.DataFrame,
    quotes: pd.DataFrame,
    base_ccy: str = "PLN",
    *,
    analytics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    # Reuse the caller's transaction summary when it has already been computed.
    if analytics is None:
        analytics = transaction_types_summary(transactions, base_ccy)
    positions = analytics["positions"]
    active = [{"ticker": ticker, **info} for ticker, info in positions.items() if info["quantity"] > 1e-10]
    if not active:
        return pd.DataFrame()
    pos = pd.DataFrame(active)
    if quotes.empty:
        quotes = pd.DataFrame(columns=["ticker", "name", "currency", "price", "previous_close", "day_change_pct", "price_date", "fetched_at", "source", "status", "error"])
    out = pos.merge(quotes, on="ticker", how="left")
    try:
        instruments = list_instruments()
        if not instruments.empty:
            out = out.merge(instruments[["ticker", "display_name"]], on="ticker", how="left")
            out["name"] = out["display_name"].where(out["display_name"].fillna("").str.strip() != "", out["name"])
            out = out.drop(columns=["display_name"])
    except Exception:
        pass
    out["currency"] = out["currency"].fillna("").astype(str).str.upper()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["market_value_local"] = out["quantity"] * out["price"]
    fx_info_by_currency: dict[str, dict[str, Any]] = {}
    for currency in out["currency"].dropna().astype(str).str.upper().unique():
        if not currency:
            continue
        spot_info = fx_spot_rate_info(currency, base_ccy, quotes)
        if spot_info is not None:
            fx_info_by_currency[currency] = spot_info
        else:
            reference_rate = fx_rate(currency, base_ccy)
            fx_info_by_currency[currency] = {
                "rate": reference_rate, "previous_rate": reference_rate,
                "returned_date": fx_returned_date(currency, base_ccy), "fetched_at": None,
                "ticker": "", "kind": "reference",
            }
    out["fx_to_base"] = out["currency"].map(lambda c: fx_info_by_currency.get(str(c).upper(), {}).get("rate"))
    out["fx_previous_to_base"] = out["currency"].map(lambda c: fx_info_by_currency.get(str(c).upper(), {}).get("previous_rate"))
    out["fx_source"] = out["currency"].map(lambda c: fx_info_by_currency.get(str(c).upper(), {}).get("kind", "reference"))
    out["market_value_base"] = out["market_value_local"] * out["fx_to_base"]
    out["unrealized_pl_base"] = out.apply(lambda r: r["market_value_base"] - r["cost_basis_base"] if pd.notna(r["market_value_base"]) and r["cost_basis_known"] and r["cost_basis_base"] is not None else np.nan, axis=1)
    out["unrealized_pl_pct"] = out.apply(lambda r: r["unrealized_pl_base"] / r["cost_basis_base"] * 100 if pd.notna(r["unrealized_pl_base"]) and r["cost_basis_base"] > 0 else np.nan, axis=1)
    total = out["market_value_base"].sum(min_count=1)
    valuation_complete = bool(out["market_value_base"].notna().all())
    out["weight_pct"] = out["market_value_base"] / total * 100 if valuation_complete and pd.notna(total) and total > 0 else np.nan
    previous_price_values = out["previous_close"] if "previous_close" in out else pd.Series(np.nan, index=out.index)
    previous_price = pd.to_numeric(previous_price_values, errors="coerce")
    previous_fx = pd.to_numeric(out["fx_previous_to_base"], errors="coerce").fillna(out["fx_to_base"])
    full_day_change = out["quantity"] * (previous_price * previous_fx - out["price"] * out["fx_to_base"])
    out["day_change_base"] = -full_day_change
    price_only_day_change = out["market_value_base"] * pd.to_numeric(out["day_change_pct"], errors="coerce") / 100
    out["day_change_base"] = out["day_change_base"].where(previous_price.notna(), price_only_day_change)
    out.attrs["analytics"] = analytics
    out.attrs["valuation_complete"] = valuation_complete
    return out


def portfolio_metrics(portfolio: pd.DataFrame, base_ccy: str = "PLN") -> dict[str, Any]:
    if portfolio.empty:
        return {"value": 0.0, "known_value": 0.0, "valuation_complete": True, "unrealized_pl": None, "unrealized_pl_pct": None,
                "unrealized_pl_complete": False, "largest": "—", "realized_pl": 0.0, "realized_pl_complete": True, "dividends": 0.0, "fees": 0.0}
    known_value = portfolio["market_value_base"].sum(min_count=1)
    valuation_complete = bool(portfolio["market_value_base"].notna().all())
    all_costs_known = bool(portfolio["unrealized_pl_base"].notna().all())
    valid = portfolio.dropna(subset=["unrealized_pl_base", "cost_basis_base"])
    basis_sum = valid["cost_basis_base"].sum() if len(valid) else None
    pl_sum = valid["unrealized_pl_base"].sum() if len(valid) else None
    weights = portfolio.dropna(subset=["market_value_base"]).sort_values("market_value_base", ascending=False)
    analytics = portfolio.attrs.get("analytics", {})
    realized_known = bool(analytics.get("realized_pl_known", True))
    return {"value": float(known_value) if valuation_complete and pd.notna(known_value) else None,
        "known_value": float(known_value) if pd.notna(known_value) else 0.0,
        "valuation_complete": valuation_complete,
        "unrealized_pl": float(pl_sum) if pl_sum is not None and all_costs_known else None,
        "unrealized_pl_pct": float(pl_sum / basis_sum * 100) if pl_sum is not None and all_costs_known and basis_sum else None,
        "unrealized_pl_complete": all_costs_known,
        "largest": str(weights.iloc[0]["ticker"]) if len(weights) and valuation_complete else "—",
        "realized_pl": float(analytics.get("realized_pl", 0)) if realized_known else None,
        "realized_pl_complete": realized_known,
        "dividends": float(analytics.get("dividends", 0)),
        "fees": float(analytics.get("fees", 0))}


def save_snapshot(portfolio: pd.DataFrame, base_currency: str = "PLN", note: str = "") -> None:
    initialize_db()
    value = float(portfolio["market_value_base"].sum()) if not portfolio.empty and portfolio["market_value_base"].notna().all() else None
    as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not portfolio.empty and value is None:
        note = (note + "; " if note else "") + "Incomplete valuation; one or more price/FX values missing"
    count = int(portfolio["price"].notna().sum()) if not portfolio.empty else 0
    with _connect() as con:
        con.execute("INSERT INTO portfolio_snapshots(as_of,base_currency,market_value,quote_count,note) VALUES(?,?,?,?,?)",
                    (as_of, base_currency, value, count, note))


def read_snapshots(base_currency: str = "PLN", limit: int = 500) -> pd.DataFrame:
    initialize_db()
    with _connect() as con:
        rows = con.execute("SELECT as_of,base_currency,market_value,quote_count,note FROM portfolio_snapshots WHERE base_currency=? ORDER BY as_of DESC LIMIT ?",
                           (base_currency, int(limit))).fetchall()
    if not rows:
        return pd.DataFrame(columns=["as_of", "base_currency", "market_value", "quote_count", "note"])
    return pd.DataFrame([dict(r) for r in rows]).sort_values("as_of")


def _period_days(period: str) -> int | None:
    mapping = {"1mo": 45, "3mo": 110, "6mo": 220, "1y": 400, "2y": 770, "3y": 1135,
               "5y": 1870, "10y": 3700}
    return mapping.get(str(period).lower())


def _cached_history(symbols: list[str], period: str) -> pd.DataFrame:
    initialize_db()
    days = _period_days(period)
    start = (date.today() - pd.Timedelta(days=days)).isoformat() if days else "1970-01-01"
    placeholders = ",".join("?" for _ in symbols)
    with _connect() as con:
        rows = con.execute(f"SELECT ticker,price_date,adjusted_close FROM market_prices WHERE ticker IN ({placeholders}) AND price_date>=? ORDER BY price_date",
                            (*symbols, start)).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows])
    frame["price_date"] = pd.to_datetime(frame["price_date"], errors="coerce")
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    frame = frame.dropna(subset=["price_date", "adjusted_close"])
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot_table(index="price_date", columns="ticker", values="adjusted_close", aggfunc="last").sort_index()


def get_history_fetch_times(tickers: Iterable[str]) -> dict[str, str]:
    """Return the latest cache write time for each ticker's adjusted price history."""
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not symbols:
        return {}
    initialize_db()
    placeholders = ",".join("?" for _ in symbols)
    with _connect() as con:
        rows = con.execute(
            f"SELECT ticker,MAX(fetched_at) AS fetched_at FROM market_prices WHERE ticker IN ({placeholders}) GROUP BY ticker",
            symbols,
        ).fetchall()
    return {str(row["ticker"]): str(row["fetched_at"] or "") for row in rows}


def get_history(tickers: Iterable[str], period: str = "1y", *, force_refresh: bool = False) -> pd.DataFrame:
    """Return adjusted-close history with persistent SQLite caching.

    Missing dates stay missing; this function deliberately does not forward-fill
    each ETF forever because that can manufacture zero returns and suppress risk.
    The analysis engine may apply a small, explicit fill only for calendar/FX joins.
    """
    symbols = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not symbols:
        return pd.DataFrame()
    initialize_db()
    cached = _cached_history(symbols, period)
    days = _period_days(period)
    start = pd.Timestamp(date.today() - pd.Timedelta(days=days)).normalize() if days else pd.Timestamp("1970-01-01")
    cache_complete = bool(not cached.empty and set(symbols).issubset(set(map(str, cached.columns))))
    if cache_complete:
        cache_recent = True
        has_window = True
        for symbol in symbols:
            points = pd.to_numeric(cached[symbol], errors="coerce").dropna()
            if points.empty:
                cache_recent = False
                has_window = False
                continue
            if (pd.Timestamp(date.today()) - pd.Timestamp(points.index.max()).normalize()).days > 5:
                cache_recent = False
            # If every symbol has fewer than 20 daily observations, an ETF may be
            # newly listed; still refresh until that short history is established.
            if period != "max" and points.index.min().normalize() > start + pd.Timedelta(days=10) and len(points) < 20:
                has_window = False
        if not force_refresh and cache_recent and has_window:
            return cached.reindex(columns=symbols)
    if yf is None:
        return cached.reindex(columns=[s for s in symbols if s in cached.columns]) if not cached.empty else pd.DataFrame()
    try:
        data = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                data = yf.download(symbols, period=period, auto_adjust=True, progress=False, group_by="ticker", threads=True)
                if not data.empty:
                    break
            except Exception as exc:
                last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
        if data.empty:
            # Keep any cached historical series visible after a provider outage.
            return cached.reindex(columns=[s for s in symbols if s in cached.columns]) if not cached.empty else pd.DataFrame()
        result: dict[str, pd.Series] = {}
        if len(symbols) == 1:
            symbol = symbols[0]
            if isinstance(data.columns, pd.MultiIndex):
                close = data[(symbol, "Close")] if (symbol, "Close") in data.columns else None
            else:
                close = data["Close"] if "Close" in data.columns else None
            if close is not None:
                result[symbol] = close.dropna()
        else:
            for symbol in symbols:
                try:
                    result[symbol] = data[(symbol, "Close")].dropna()
                except Exception:
                    continue
        if not result:
            return cached.reindex(columns=[s for s in symbols if s in cached.columns]) if not cached.empty else pd.DataFrame()
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _connect() as con:
            for symbol, series in result.items():
                clean = pd.to_numeric(series, errors="coerce").dropna()
                for idx, value in clean.items():
                    try:
                        dt = pd.Timestamp(idx)
                        if dt.tzinfo is not None:
                            dt = dt.tz_convert(None)
                        price_date = dt.normalize().date().isoformat()
                        number = float(value)
                        if not np.isfinite(number) or number <= 0:
                            continue
                        con.execute("""INSERT INTO market_prices(ticker,price_date,adjusted_close,fetched_at)
                            VALUES(?,?,?,?) ON CONFLICT(ticker,price_date) DO UPDATE SET
                            adjusted_close=excluded.adjusted_close,fetched_at=excluded.fetched_at""",
                            (symbol, price_date, number, stamp))
                    except (TypeError, ValueError, OverflowError):
                        continue
        refreshed = _cached_history(symbols, period)
        if not refreshed.empty:
            return refreshed.reindex(columns=symbols)
        # Fallback to the live provider frame if the cache write was unavailable.
        return pd.DataFrame(result).sort_index().reindex(columns=symbols)
    except Exception:
        return cached.reindex(columns=[s for s in symbols if s in cached.columns]) if not cached.empty else pd.DataFrame()

def simulate_forecast(current_value: float, monthly_contribution: float, annual_return: float,
                      annual_volatility: float, inflation: float, horizons: tuple[int, ...] = (1, 3, 5, 10),
                      simulations: int = 4000, seed: int | None = 42,
                      reporting_currency: str = "PLN") -> pd.DataFrame:
    """Monte Carlo illustration using a simple lognormal return assumption, not a prediction."""
    current_value = max(0.0, float(current_value))
    monthly_contribution = max(0.0, float(monthly_contribution))
    annual_volatility = max(0.0, float(annual_volatility))
    annual_return = float(annual_return)
    inflation = float(inflation)
    max_years = max(horizons)
    rng = np.random.default_rng(seed)
    monthly_sigma = annual_volatility / np.sqrt(12)
    monthly_log_drift = (np.log1p(annual_return) - 0.5 * annual_volatility**2) / 12
    log_returns = rng.normal(monthly_log_drift, monthly_sigma, size=(simulations, max_years * 12))
    returns = np.exp(np.clip(log_returns, -3, 3))
    values = np.full(simulations, current_value, dtype=float)
    output: list[dict[str, Any]] = []
    horizon_set = set(horizons)
    for month in range(1, max_years * 12 + 1):
        values = values * returns[:, month - 1] + monthly_contribution
        if month % 12 == 0 and month // 12 in horizon_set:
            years = month // 12
            p10, p50, p90 = np.percentile(values, [10, 50, 90])
            def real(v: float) -> float:
                return float(v / ((1 + inflation) ** years))
            output.append({"Horizon (years)": years,
                "Contributions added": round(monthly_contribution * 12 * years, 2),
                "10th percentile (nominal)": round(float(p10), 2),
                "Median (nominal)": round(float(p50), 2),
                "90th percentile (nominal)": round(float(p90), 2),
                f"10th percentile (today's {reporting_currency})": round(real(float(p10)), 2),
                f"Median (today's {reporting_currency})": round(real(float(p50)), 2),
                f"90th percentile (today's {reporting_currency})": round(real(float(p90)), 2)})
    return pd.DataFrame(output)
