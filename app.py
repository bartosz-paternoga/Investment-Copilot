from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from database import database_url, set_active_tenant_schema
from auth import render_auth_gate, render_account_sidebar

from portfolio import (
    add_transaction, add_watchlist_ticker, build_portfolio, delete_transaction, fx_rate, fx_rate_info, get_history, get_quotes,
    fx_spot_pair_tickers, fx_spot_rate_info, save_watchlist_order,
    initialize_db, list_instruments, list_transactions, list_watchlist, portfolio_metrics, read_snapshots,
    remove_watchlist_ticker, save_instrument, save_snapshot, simulate_forecast, transaction_types_summary,
    replace_quick_position, remove_ticker_from_portfolio,
    resolve_position_currency,
    get_fx_history, get_history_fetch_times, save_etf_holdings, list_etf_holdings, read_market_data_runs, market_data_cache_summary,
)
from advanced_analytics import historical_risk_table, portfolio_xirr
from performance_engine import (
    compare_to_benchmark, convert_prices_to_currency, linked_attribution,
    performance_summary, portfolio_total_return,
)
from research_analytics import calculate_etf_overlap, holdings_weight_coverage

load_dotenv()
try:
    APP_VERSION = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()
except OSError:
    APP_VERSION = "dev"
st.set_page_config(page_title="Investment Copilot | Portfolio", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

_auth_transition_overlay = st.empty()
_auth_transition_overlay_visible = bool(
    st.session_state.get("_auth_transition_loading", False)
)


def _clear_auth_transition_overlay() -> None:
    global _auth_transition_overlay_visible
    if _auth_transition_overlay_visible:
        _auth_transition_overlay.empty()
    _auth_transition_overlay_visible = False
    st.session_state["_auth_transition_loading"] = False


def _show_auth_transition_overlay() -> None:
    global _auth_transition_overlay_visible
    _auth_transition_overlay_visible = True
    _auth_transition_overlay.markdown(
        """
        <style>
        #auth-transition-overlay {
            position: fixed;
            inset: 0;
            z-index: 2147483647;
            display: grid;
            place-content: center;
            justify-items: center;
            gap: 14px;
            background: #f4f6f8;
            color: #475569;
            font: 600 16px/1.4 sans-serif;
        }
        #auth-transition-spinner {
            width: 34px;
            height: 34px;
            border: 3px solid #d9e2eb;
            border-top-color: #64748b;
            border-radius: 50%;
            animation: auth-transition-spin .8s linear infinite;
        }
        @keyframes auth-transition-spin {
            to { transform: rotate(360deg); }
        }
        </style>
        <div id="auth-transition-overlay" role="status" aria-live="polite">
            <div id="auth-transition-spinner"></div>
            <div>Opening your portfolio…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if _auth_transition_overlay_visible:
    _show_auth_transition_overlay()

# A restrained steel/slate palette keeps charts consistent with the neutral UI.
px.defaults.color_discrete_sequence = ["#475569", "#94a3b8", "#cbd5e1", "#334155", "#64748b", "#e2e8f0", "#7c8798", "#b0bac7"]

# Install theme and component styles before rendering any UI elements.
st.markdown("""
<style>
:root {
  --adatch-ink: #17212f;
  --adatch-muted: #687586;
  --adatch-line: #e1e6ec;
  --adatch-surface: #ffffff;
  --adatch-canvas: #f4f6f8;
  --adatch-steel: #64748b;
  --adatch-steel-soft: #edf1f5;
  --adatch-red: #d9545d;
  --adatch-steel-deep: #475569;
  --adatch-steel-hover: #334155;
}
.stApp { background: var(--adatch-canvas); color: var(--adatch-ink); }
[data-testid="stHeader"] { background: rgba(244,246,248,.96); }
[data-testid="stMainBlockContainer"] { max-width: 1480px; padding-top: 1.65rem; padding-bottom: 3.5rem; }
[data-testid="stAppViewContainer"] { background: var(--adatch-canvas); }
.dev-mode-notice { display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; padding:13px 16px; margin:0 0 22px; border:1px solid #cbd5e1; border-left:4px solid #64748b; border-radius:12px; background:#eef2f6; color:#273449 !important; opacity:1 !important; font-size:.9rem; line-height:1.55; box-shadow:0 2px 8px rgba(30,41,59,.04); }
.dev-mode-notice, .dev-mode-notice * { color:#273449 !important; opacity:1 !important; -webkit-text-fill-color:#273449 !important; }
.dev-mode-notice strong { font-weight:800; }
.dev-mode-notice span { font-weight:500; }
[data-testid="stSidebar"] { background: #fbfcfd; border-right: 1px solid var(--adatch-line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--adatch-muted); }
[data-testid="stSidebar"] hr { border-color: var(--adatch-line); margin: 1rem 0; }
.brand-lockup { display:flex; align-items:center; gap:11px; padding: 0 0 1.25rem 0; margin-bottom: .45rem; border-bottom:1px solid var(--adatch-line); }
.brand-mark { width:39px; height:39px; display:flex; align-items:center; justify-content:center; color:white; background:linear-gradient(145deg,#64748b,#334155); border-radius:12px; font-weight:800; font-size:20px; box-shadow:0 5px 12px rgba(51,65,85,.16); }
.brand-name { color:var(--adatch-ink); font-weight:760; font-size:23px; letter-spacing:-.8px; line-height:1.1; }
.brand-caption { margin-top:4px; font-size:9px; letter-spacing:.14em; color:#94a0ad; font-weight:700; }
.terminal-tag { color: #64748b; font-size: 10px; letter-spacing: .16em; text-transform: uppercase; font-weight: 750; margin-bottom: 9px; }
h1, h2, h3 { color: var(--adatch-ink); letter-spacing: -.035em; }
h1 { font-size: 2rem !important; font-weight: 740 !important; }
h2 { font-size: 1.4rem !important; font-weight: 710 !important; }
h3 { font-size: 1.1rem !important; font-weight: 690 !important; }
p, li { color: #4b5868; }
.small-muted { color:var(--adatch-muted); font-size:.84rem; }
.portfolio-hero { display:flex; justify-content:space-between; align-items:center; gap:20px; padding:3px 1px 20px; }
.hero-subtitle { color: #647184; font-size: .98rem; line-height:1.55; margin-top:5px; max-width:620px; overflow-wrap:anywhere; }
.portfolio-hero > div:first-child { min-width:0; flex:1 1 420px; }
.portfolio-hero h1 { overflow-wrap:anywhere; }
.hero-badge { white-space:nowrap; color:#475569; background:var(--adatch-steel-soft); border:1px solid #d8e0e8; padding:8px 12px; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.035em; }
.section-heading { color:var(--adatch-ink); font-size:1.12rem; font-weight:730; margin:16px 0 9px; display:flex; align-items:baseline; gap:10px; letter-spacing:-.025em; }
.holdings-heading, .action-heading { margin-top:24px; }
.empty-panel { min-height:245px; border:1px dashed #d8e0e8; border-radius:16px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:26px; color:var(--adatch-ink); background:rgba(255,255,255,.75); }
.empty-panel p { color:var(--adatch-muted); max-width:330px; margin-bottom:0; }
.empty-icon { font-size:30px; color:var(--adatch-steel); margin-bottom:10px; }
[data-testid="stMetric"] { background:var(--adatch-surface); border:1px solid var(--adatch-line); padding:17px 18px; border-radius:16px; min-height:112px; box-shadow:0 4px 16px rgba(30,41,59,.035); }
[data-testid="stMetric"] label { color:var(--adatch-muted) !important; font-size:.82rem !important; font-weight:600 !important; }
[data-testid="stMetricValue"] { color:var(--adatch-ink); font-size:clamp(1.25rem, 1.6vw, 1.85rem) !important; font-weight:750 !important; letter-spacing:-.04em; }
[data-testid="stMetricDelta"] { font-weight:650; font-size:.8rem; }
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid var(--adatch-line); border-radius:16px; padding:10px 8px 1px; box-shadow:0 4px 16px rgba(30,41,59,.03); }
div[data-testid="stDataFrame"] { border:1px solid var(--adatch-line); border-radius:14px; overflow:hidden; background:#fff; }
[data-testid="stExpander"] { border:1px solid var(--adatch-line); border-radius:13px; background:#fff; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--adatch-line); }
[data-testid="stTabs"] button[data-baseweb="tab"] { color:#687588; font-weight:650; border-radius:9px 9px 0 0; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#334155; border-bottom-color:#64748b; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:5px; }
[data-testid="stSidebar"] [role="radiogroup"] label { border-radius:10px; padding:8px 10px; margin:0; color:#586576; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#edf1f5; color:#334155; font-weight:700; }
[data-testid="stSidebar"] [role="radiogroup"] label p { color:inherit; }
/* Component styles are intentionally explicit: Streamlit's theme defaults vary by version. */
.stButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button { border-radius:11px !important; font-weight:650 !important; min-height:2.75rem; border:1px solid #d5dde6 !important; transition:background .15s ease, border-color .15s ease, transform .15s ease; box-shadow:none !important; }
.stButton > button:not([kind*="primary"]):not([data-testid*="primary"]), [data-testid="stDownloadButton"] button { background:#ffffff !important; color:#263445 !important; border-color:#d9e2e9 !important; }
.stButton > button:not([kind*="primary"]):not([data-testid*="primary"]):hover, [data-testid="stDownloadButton"] button:hover { background:#f1f4f7 !important; color:#334155 !important; border-color:#94a3b8 !important; }
.stButton > button[kind*="primary"], .stButton > button[data-testid*="primary"], [data-testid="stFormSubmitButton"] button[kind*="primary"], [data-testid="stFormSubmitButton"] button[data-testid*="primary"], button[data-testid="stBaseButton-primary"] { background:#64748b !important; background-color:#64748b !important; color:#ffffff !important; border-color:#64748b !important; -webkit-text-fill-color:#ffffff !important; }
.stButton > button[kind*="primary"] *, .stButton > button[data-testid*="primary"] *, [data-testid="stFormSubmitButton"] button[kind*="primary"] *, [data-testid="stFormSubmitButton"] button[data-testid*="primary"] *, button[data-testid="stBaseButton-primary"] * { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; opacity:1 !important; }
.stButton > button[kind*="primary"]:hover, .stButton > button[data-testid*="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind*="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background:#475569 !important; background-color:#475569 !important; border-color:#475569 !important; color:#ffffff !important; }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible { outline:3px solid rgba(100,116,139,.32) !important; outline-offset:2px; }
.stButton > button:disabled, [data-testid="stFormSubmitButton"] button:disabled { background:#edf1f4 !important; color:#8a96a3 !important; border-color:#e1e7ec !important; -webkit-text-fill-color:#8a96a3 !important; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stTextArea"] textarea { border-radius:10px; }
[data-testid="stAlert"] { border-radius:12px; }
[data-testid="stCaptionContainer"] { color:#8591a0; }
hr { border-color:var(--adatch-line); }
@media (max-width: 850px) {
  [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
  .portfolio-hero { align-items:flex-start; flex-direction:column; gap:12px; }
  .portfolio-hero > div:first-child { flex-basis:auto; width:100%; }
  .portfolio-hero .hero-badge { white-space:normal; }
  .hero-badge { align-self:flex-start; }
}
</style>
""", unsafe_allow_html=True)

# Hosted mode is fail-closed: DATABASE_URL is mandatory and all database access
# follows a freshly validated user's tenant schema. SQLite is opt-in for local dev only.
_current_user = None
if database_url():
    try:
        _current_user = render_auth_gate(st, show_transition=_show_auth_transition_overlay)
    except Exception:
        _clear_auth_transition_overlay()
        st.error("The managed PostgreSQL database could not be reached or initialized. Check DATABASE_URL, database availability, and required permissions in Streamlit Secrets.")
        st.stop()
    if _current_user is None:
        if not st.session_state.get("_auth_cookie_probe_pending", False):
            _clear_auth_transition_overlay()
        st.stop()
    set_active_tenant_schema(_current_user["tenant_schema"])
    initialize_db()
elif os.getenv("ETF_APP_LOCAL_DEV_MODE", "").strip().lower() in {"1", "true", "yes"}:
    st.markdown('<div class="dev-mode-notice"><strong>Local development mode</strong><span>Authentication is bypassed and data is stored in local SQLite. This mode is for testing on your own computer only.</span></div>', unsafe_allow_html=True)
    set_active_tenant_schema(None)
    initialize_db()
else:
    st.error("Deployment is not configured. Add DATABASE_URL and the bootstrap administrator secrets in Streamlit Community Cloud. For local single-user development only, set ETF_APP_LOCAL_DEV_MODE=1.")
    st.stop()



PAGES = ["Overview", "Investments", "Transactions", "Performance", "ETF Research", "Watchlist", "Holdings & risk", "Market analysis", "AI Analyst", "Forecast Lab", "Data operations", "Methodology & data"]
st.sidebar.markdown('''<div class="brand-lockup"><div class="brand-mark">IP</div><div><div class="brand-name">Investment Copilot</div><div class="brand-caption">INVESTOR PORTFOLIO</div></div></div>''', unsafe_allow_html=True)
st.sidebar.caption(f"Version {APP_VERSION}")
page = st.sidebar.radio("Workspace", PAGES, index=0, label_visibility="collapsed", key="workspace_page")
base_ccy = st.sidebar.selectbox("Reporting currency", ["PLN", "EUR", "USD", "GBP"], index=0)
transactions = list_transactions()
# Transaction analytics can involve historical FX lookups. Cache it within this
# user's Streamlit session and invalidate it whenever the ledger or currency changes.
# The user ID in the key prevents reuse across accounts in the same browser session.
ledger_fingerprint = (
    pd.util.hash_pandas_object(transactions, index=True).values.tobytes()
    if not transactions.empty else b""
)
analytics_cache_key = (
    str(_current_user["user_id"]) if _current_user is not None else "local",
    base_ccy,
    ledger_fingerprint,
)
_cached_analytics = st.session_state.get("_transaction_analytics_cache")
if _cached_analytics is not None and _cached_analytics[0] == analytics_cache_key:
    analytics = _cached_analytics[1]
else:
    analytics = transaction_types_summary(transactions, base_ccy)
    st.session_state["_transaction_analytics_cache"] = (analytics_cache_key, analytics)
active_tickers = sorted([t for t, p in analytics["positions"].items() if p["quantity"] > 1e-10])
watchlist_tickers = list_watchlist().get("ticker", pd.Series(dtype=str)).astype(str).tolist()
primary_quote_tickers = sorted(set(active_tickers + watchlist_tickers))
cash_currencies = sorted({str(currency).upper() for currency in analytics.get("cash", {}) if str(currency).strip()})
user_key = str(_current_user["user_id"]) if _current_user is not None else "local"
quote_bundle_key = (user_key, base_ccy, tuple(primary_quote_tickers), tuple(cash_currencies))


def load_current_market_quotes(force_refresh: bool) -> pd.DataFrame:
    """Load holdings/watchlist prices plus spot FX pairs needed for valuation."""
    primary = get_quotes(primary_quote_tickers, force_refresh=force_refresh)
    currencies = set(cash_currencies)
    if not primary.empty and "currency" in primary:
        currencies.update(primary["currency"].fillna("").astype(str).str.upper())
    bundle = primary
    missing_spot = {currency for currency in currencies
                    if currency and fx_spot_rate_info(currency, base_ccy, bundle) is None}
    direct_tickers = [symbol for symbol in fx_spot_pair_tickers(missing_spot, base_ccy)
                      if symbol not in set(primary_quote_tickers)]
    direct_quotes = get_quotes(direct_tickers, force_refresh=force_refresh)
    if not direct_quotes.empty:
        bundle = pd.concat([bundle, direct_quotes], ignore_index=True).drop_duplicates("ticker", keep="last")
    missing_spot = {currency for currency in missing_spot
                    if fx_spot_rate_info(currency, base_ccy, bundle) is None}
    inverse_tickers = [symbol for symbol in fx_spot_pair_tickers(missing_spot, base_ccy, inverse=True)
                       if symbol not in set(primary_quote_tickers)]
    inverse_quotes = get_quotes(inverse_tickers, force_refresh=force_refresh)
    if not inverse_quotes.empty:
        bundle = pd.concat([bundle, inverse_quotes], ignore_index=True).drop_duplicates("ticker", keep="last")
    for currency in currencies:
        if fx_spot_rate_info(currency, base_ccy, bundle) is None:
            fx_rate_info(currency, base_ccy, force_refresh=force_refresh)
    return bundle

if "quotes" not in st.session_state:
    st.session_state.quotes = None
if "quotes_for" not in st.session_state:
    st.session_state.quotes_for = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = None

session_token = str(st.session_state.get("auth_session_token", ""))
login_refresh_key = hashlib.sha256(session_token.encode("utf-8")).hexdigest() if session_token else user_key
refresh_on_login = st.session_state.get("_market_data_login_key") != login_refresh_key


def force_market_refresh(note: str) -> None:
    with st.spinner("Refreshing prices and current FX quotes for your holdings and watchlist…"):
        st.session_state.quotes = load_current_market_quotes(force_refresh=True)
        st.session_state.quotes_for = quote_bundle_key
        st.session_state.last_refresh = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        st.session_state._market_data_login_key = login_refresh_key
        report = build_portfolio(transactions, st.session_state.quotes, base_ccy, analytics=analytics)
        save_snapshot(report, base_ccy, note)

with st.sidebar:
    st.markdown("---")
    st.caption(f"Ledger: {len(transactions)} transactions · {len(active_tickers)} open positions")
    if st.button("↻ Refresh prices and FX", use_container_width=True,
                 help="Updates prices for your holdings and watchlist, plus current currency quotes."):
        force_market_refresh("Manual refresh")
        st.rerun()
    st.markdown("---")
    st.caption("Current prices and FX update at sign-in. Use this button to refresh them on any page; historical analyses refresh their daily data when recalculated. No trades are executed.")

if _current_user is not None:
    render_account_sidebar(st, _current_user)

# Refresh the current market bundle once for each authenticated browser session.
# The same bundle serves holdings and watchlist tables across every page.
if refresh_on_login or st.session_state.quotes is None or st.session_state.quotes_for != quote_bundle_key:
    if primary_quote_tickers or cash_currencies:
        with st.spinner("Updating latest available prices and current FX quotes…" if refresh_on_login else "Loading latest available prices and FX…"):
            st.session_state.quotes = load_current_market_quotes(force_refresh=refresh_on_login)
            st.session_state.quotes_for = quote_bundle_key
            st.session_state.last_refresh = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    else:
        st.session_state.quotes = pd.DataFrame()
        st.session_state.quotes_for = quote_bundle_key
    if refresh_on_login:
        st.session_state._market_data_login_key = login_refresh_key

quotes = st.session_state.quotes if isinstance(st.session_state.quotes, pd.DataFrame) else pd.DataFrame()
try:
    display_timezone = str(st.context.timezone or "UTC")
except (AttributeError, RuntimeError):
    display_timezone = "UTC"


def local_market_time(value) -> str:
    """Format a UTC market-data fetch timestamp in the viewer's browser timezone."""
    if value is None or not str(value).strip():
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    try:
        return parsed.tz_convert(display_timezone).strftime("%Y-%m-%d %H:%M %Z")
    except (TypeError, ValueError):
        return parsed.strftime("%Y-%m-%d %H:%M UTC")


def market_timing_label(source_date, fetched_local) -> str:
    """Combine a provider's source date with its local fetch time for table cells."""
    source_date = str(source_date or "").strip()
    fetched_local = str(fetched_local or "").strip()
    if source_date and fetched_local:
        parts = fetched_local.split(" ", 2)
        fetch_time = " ".join(parts[1:]) if len(parts) == 3 and parts[0] == source_date else fetched_local
        return f"{source_date} · {fetch_time}"
    return source_date or fetched_local or "—"


def add_quote_timing_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add a compact date/fetch-time cell to a quote table."""
    if data is None or data.empty:
        return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
    result = data.copy()
    if "fetched_at" in result.columns:
        result["quote_fetched_local"] = result["fetched_at"].map(local_market_time)
        result["price_timing"] = result.apply(
            lambda row: market_timing_label(row.get("price_date"), row.get("quote_fetched_local")), axis=1
        )
    return result


def current_fx_info(currency: str) -> dict:
    """Use the current quote pair for daily values; fall back to a daily reference rate."""
    currency = str(currency or "").upper().strip()
    if currency == base_ccy:
        return {"rate": 1.0, "returned_date": "—", "fetched_at": None, "kind": "same currency"}
    spot = fx_spot_rate_info(currency, base_ccy, quotes)
    if spot is not None:
        return spot
    reference = fx_rate_info(currency, base_ccy) or {}
    if reference:
        reference["kind"] = "reference"
    return reference


def fx_timing_label(info: dict) -> str:
    timing = market_timing_label(
        info.get("returned_date", "Unavailable"), local_market_time(info.get("fetched_at"))
    )
    kind = str(info.get("kind", "reference"))
    if kind == "spot":
        return f"Spot · {timing}"
    if kind == "reference":
        return f"Daily ref · {timing}"
    return timing


quotes = add_quote_timing_columns(quotes)
portfolio = build_portfolio(transactions, quotes, base_ccy, analytics=analytics)
if not portfolio.empty:
    fx_metadata = {}
    for currency in sorted(set(portfolio["currency"].fillna("").astype(str).str.upper())):
        if not currency:
            continue
        fx_metadata[currency] = current_fx_info(currency)
    portfolio["fx_timing"] = portfolio["currency"].map(
        lambda currency: fx_timing_label(fx_metadata.get(str(currency or "").upper(), {}))
    )
market_data_status_tickers = set(active_tickers)
for currency in set(portfolio.get("currency", pd.Series(dtype=str)).fillna("").astype(str).str.upper()) | set(cash_currencies):
    info = current_fx_info(currency)
    if info.get("ticker"):
        market_data_status_tickers.add(info["ticker"])
market_data_status = quotes[quotes["ticker"].isin(market_data_status_tickers)] if "ticker" in quotes else pd.DataFrame()
metrics = portfolio_metrics(portfolio, base_ccy)
_clear_auth_transition_overlay()


def european_number(value, decimals=2, suffix="", signed=False):
    """Format a number with European decimal commas and space thousands groups."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    specifier = f"{'+,' if signed else ','}.{int(decimals)}f"
    rendered = format(number, specifier).replace(",", "\0").replace(".", ",").replace("\0", " ")
    return f"{rendered}{suffix}"


def european_percent(value, signed=False, decimals=2):
    return european_number(value, decimals=decimals, suffix="%", signed=signed)


def european_dataframe(data, *args, **kwargs):
    """Render read-only tables with numeric values in the requested European format."""
    global _render_dataframe
    config = kwargs.get("column_config") or {}
    display_config = dict(config)
    formatters = {}
    if isinstance(data, pd.DataFrame) and not data.empty:
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column].dtype):
                continue
            column_name = str(column)
            column_config = config.get(column_name, {})
            type_config = column_config.get("type_config", {}) if isinstance(column_config, dict) else {}
            old_format = str(type_config.get("format") or "")
            normalized_name = column_name.casefold().strip()
            percentage_column = "%%" in old_format or "%" in column_name
            integer_column = (
                (pd.api.types.is_integer_dtype(data[column].dtype) and not percentage_column)
                or "%.0f" in old_format or "%d" in old_format
                or normalized_name in {"units", "quantity", "order", "rank", "#", "years"}
                or "observations" in normalized_name or normalized_name.endswith(" count")
            )
            signed_column = "%+" in old_format
            decimals = 0 if integer_column else 2
            suffix = "%" if percentage_column else ""
            formatters[column_name] = lambda value, d=decimals, s=suffix, sign=signed_column: european_number(
                value, decimals=d, suffix=s, signed=sign
            )
            if isinstance(column_config, dict) and type_config.get("type") == "number":
                clean_config = dict(column_config)
                clean_type_config = dict(type_config)
                clean_type_config.pop("format", None)
                clean_config["type_config"] = clean_type_config
                display_config[column_name] = clean_config
        if formatters:
            data = data.style.format(formatters, na_rep="—")
    kwargs["column_config"] = display_config
    return _render_dataframe(data, *args, **kwargs)


_render_dataframe = st.dataframe


def money(value, currency=base_ccy, signed=False):
    amount = european_number(value, decimals=2, signed=signed)
    return "—" if amount == "—" else f"{amount} {currency}"


_xirr_fx_series_cache = {}


def historical_fx_for_xirr(source: str, target: str, on_date: str):
    """Use one cached historical FX range per currency pair when calculating XIRR."""
    source, target = str(source or "").upper(), str(target or "").upper()
    if source == target:
        return 1.0
    key = (source, target)
    if key not in _xirr_fx_series_cache:
        dates = pd.to_datetime(transactions.get("trade_date", pd.Series(dtype=str)), errors="coerce").dropna()
        if dates.empty:
            _xirr_fx_series_cache[key] = pd.Series(dtype=float)
        else:
            _xirr_fx_series_cache[key] = get_fx_history(source, target, dates.min().date().isoformat(), dates.max().date().isoformat())
    series = _xirr_fx_series_cache[key]
    if not series.empty:
        try:
            date_key = pd.Timestamp(str(on_date)[:10])
            index = series.sort_index()
            position = int(index.index.searchsorted(date_key, side="right")) - 1
            if position >= 0:
                rate_date = pd.Timestamp(index.index[position])
                rate = float(index.iloc[position])
                if (date_key - rate_date).days <= 7 and np.isfinite(rate) and rate > 0:
                    return rate
        except (ValueError, TypeError, IndexError):
            pass
    return fx_rate(source, target, on_date)


def display_position_table(data: pd.DataFrame):
    if data.empty:
        st.info("There are no open ETF positions yet. Record an OPENING position or BUY transaction in Transactions.")
        return
    columns = ["ticker", "name", "quantity", "currency", "price", "day_change_pct", "price_timing", "fx_to_base", "fx_timing", "market_value_local", "market_value_base", "weight_pct", "cost_basis_base", "unrealized_pl_base", "unrealized_pl_pct", "status"]
    view = data[[c for c in columns if c in data.columns]].copy()
    view = view.rename(columns={
        "ticker":"Ticker", "name":"Instrument", "quantity":"Units", "currency":"Currency",
        "price":"Price", "price_timing":"Price date / time", "fx_to_base":f"FX → {base_ccy}",
        "fx_timing":"FX date / time", "market_value_local":"Value (CCY)",
        "market_value_base":f"Value ({base_ccy})", "weight_pct":"Weight %",
        "day_change_pct":"Price change %", "cost_basis_base":f"Cost ({base_ccy})",
        "unrealized_pl_base":f"P/L ({base_ccy})", "unrealized_pl_pct":"P/L %", "status":"Status",
    })
    european_dataframe(view, use_container_width=True, hide_index=True, column_config={
        "Ticker": st.column_config.TextColumn(width="small"),
        "Instrument": st.column_config.TextColumn(width="medium"),
        "Units": st.column_config.NumberColumn(format="%.0f", width="small"),
        "Currency": st.column_config.TextColumn(width="small"),
        "Price": st.column_config.NumberColumn(format="%.4f", width="small"),
        "Price change %": st.column_config.NumberColumn(format="%+.2f%%", width="small", help="Today's instrument price move in its listing currency; daily portfolio P/L also includes the FX move."),
        "Price date / time": st.column_config.TextColumn(width="medium", help=f"Exchange price date and quote fetch time in {display_timezone}."),
        f"FX → {base_ccy}": st.column_config.NumberColumn(format="%.5f", width="small"),
        "FX date / time": st.column_config.TextColumn(width="medium", help=f"Current spot FX quote date and fetch time in {display_timezone}; a daily reference rate is used if a spot quote is unavailable."),
        f"Value ({base_ccy})": st.column_config.NumberColumn(format="%.2f", width="small"),
        "Weight %": st.column_config.NumberColumn(format="%.2f%%", width="small"),
        "P/L %": st.column_config.NumberColumn(format="%+.2f%%", width="small"),
        f"Cost ({base_ccy})": st.column_config.NumberColumn(format="%.2f", width="small"),
        f"P/L ({base_ccy})": st.column_config.NumberColumn(format="%+.2f", width="small"),
        "Status": st.column_config.TextColumn(width="small"),
    }, row_height=34)

if page == "Overview":
    st.title("Portfolio overview")
    st.markdown('<div class="terminal-tag">TRANSACTION LEDGER · MULTI-CURRENCY VALUATION · RISK SNAPSHOT</div>', unsafe_allow_html=True)
    if transactions.empty:
        st.info("Start in **Transactions**. Add your existing holdings using an OPENING transaction, then record future purchases, sales, dividends and cash movements.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("ETF market value", money(metrics["value"]))
        c2.metric("Unrealized P/L", money(metrics["unrealized_pl"], signed=True), "Cost basis / price incomplete" if metrics["unrealized_pl"] is None else (european_percent(metrics["unrealized_pl_pct"], signed=True) if metrics["unrealized_pl_pct"] is not None else None))
        c3.metric("Realized P/L to date", money(metrics["realized_pl"], signed=True), "Incomplete cost/FX history" if not metrics["realized_pl_complete"] else None)
        c4.metric("Dividends recorded", money(metrics["dividends"]))
        if not metrics["valuation_complete"]:
            st.warning(f"Portfolio valuation is incomplete. Known priced positions sum to {money(metrics['known_value'])}, but at least one holding lacks a valid price or FX rate; the headline total is intentionally withheld.")
        if not metrics["unrealized_pl_complete"]:
            st.warning("Portfolio-wide unrealized P/L is unavailable because at least one active position has missing acquisition cost or market valuation data. Per-position figures may still be shown where valid.")
        if not metrics["realized_pl_complete"]:
            st.warning("Realized P/L is incomplete because at least one sale could not be matched to a known book cost and valid historical FX rate.")
        display_position_table(portfolio)
        status_cols = [c for c in ["ticker","currency","price","day_change_pct","price_timing","status","error"] if c in market_data_status.columns]
        if len(market_data_status) and "status" in market_data_status.columns and (market_data_status["status"] != "OK").any():
            st.warning("Some quotes are incomplete. Do not treat missing or stale prices as zero; inspect the quote status below.")
            status_view = market_data_status[status_cols].rename(columns={
                "ticker":"Ticker", "currency":"Currency", "price":"Price", "day_change_pct":"Price change %", "price_timing":"Price date / time",
                "status":"Status", "error":"Issue",
            })
            european_dataframe(status_view, use_container_width=True, hide_index=True, row_height=34, column_config={
                "Ticker":st.column_config.TextColumn(width="small"),
                "Currency":st.column_config.TextColumn(width="small"),
                "Price":st.column_config.NumberColumn(format="%.4f", width="small"),
                "Price change %":st.column_config.NumberColumn(format="%+.2f%%", width="small"),
                "Price date / time":st.column_config.TextColumn(width="medium", help=f"Exchange price date and quote fetch time in {display_timezone}."),
                "Status":st.column_config.TextColumn(width="small"),
                "Issue":st.column_config.TextColumn(width="large"),
            })
        left,right = st.columns(2)
        with left:
            st.subheader("ETF allocation")
            if not portfolio.empty and portfolio["market_value_base"].notna().any():
                fig = px.pie(portfolio.dropna(subset=["market_value_base"]), values="market_value_base", names="ticker", hole=.56)
                fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Allocation will appear when valid prices and FX rates are available.")
        with right:
            st.subheader("Contribution to today's move")
            daily = portfolio.dropna(subset=["day_change_base"]).sort_values("day_change_base") if not portfolio.empty else pd.DataFrame()
            if not daily.empty:
                fig = px.bar(daily, x="ticker", y="day_change_base", title=f"Approximate daily P/L contribution ({base_ccy})")
                fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Daily moves are unavailable for these quotes.")
        snaps = read_snapshots(base_ccy)
        if len(snaps) >= 2:
            st.subheader("Saved portfolio valuation snapshots")
            fig = px.line(snaps, x="as_of", y="market_value", markers=True, labels={"as_of":"Snapshot time", "market_value":f"ETF market value ({base_ccy})"})
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Ledger summary")
        k1,k2,k3 = st.columns(3)
        k1.metric("Fees recorded", money(metrics["fees"]))
        k2.metric("Cash contributed", money(analytics["contributions"]))
        k3.metric("Cash withdrawn", money(analytics["withdrawals"]))
        if analytics["cash"]:
            st.subheader("Cash ledger balances")
            cash_rows = []
            for currency, balance in sorted(analytics["cash"].items()):
                fx_info = current_fx_info(currency)
                rate = fx_info.get("rate") if fx_info else None
                fx_timing = fx_timing_label(fx_info or {})
                cash_rows.append({"Currency":currency,"Cash":balance,"FX rate":rate,"FX date / time":fx_timing,
                                  f"Value ({base_ccy})":balance * rate if rate is not None else np.nan})
            european_dataframe(pd.DataFrame(cash_rows), use_container_width=True, hide_index=True, row_height=34, column_config={
                "Currency":st.column_config.TextColumn(width="small"),
                "Cash":st.column_config.NumberColumn(format="%.2f", width="small"),
                "FX rate":st.column_config.NumberColumn(format="%.5f", width="small"),
                "FX date / time":st.column_config.TextColumn(width="medium", help=f"Current spot FX quote date and fetch time in {display_timezone}; a daily reference rate is used if a spot quote is unavailable."),
                f"Value ({base_ccy})":st.column_config.NumberColumn(format="%.2f", width="small"),
            })
            if any(row["Cash"] < -1e-8 for row in cash_rows):
                st.warning("One or more cash balances are negative. This commonly means buys/sells were recorded without matching cash-in/cash-out events, or the cash ledger is incomplete.")
        if analytics["missing_fx"]:
            st.warning(f"Could not resolve historical FX for {len(analytics['missing_fx'])} transaction(s). Cost basis and cumulative metrics may be incomplete. Check Methodology & data.")


elif page == "Investments":
    # A portfolio-first workspace inspired by modern consumer portfolio trackers.
    # Keep the hero and its action in normal document flow. Side-by-side
    # columns caused the refresh button to overlap/clip on narrow hosted layouts.
    st.markdown("""
    <div class="portfolio-hero">
      <div><div class="terminal-tag">YOUR INVESTMENTS · PRIVATE PORTFOLIO</div>
      <h1 style="margin:0 0 6px 0;">Investments</h1>
      <div class="hero-subtitle">Your portfolio, performance and allocation in one place.</div></div>
      <div class="hero-badge">● PORTFOLIO WORKSPACE</div>
    </div>
    """, unsafe_allow_html=True)
    action_col, _ = st.columns([1.2, 4.8])
    with action_col:
        if st.button("↻ Refresh prices and FX", type="primary", use_container_width=True, key="home_refresh_prices"):
            force_market_refresh("Manual refresh from Investments")
            st.rerun()

    open_value = metrics.get("value")
    daily_pl = (float(portfolio["day_change_base"].sum()) if not portfolio.empty and "day_change_base" in portfolio and portfolio["day_change_base"].notna().all() else None)
    daily_return_pct = None
    if daily_pl is not None and open_value is not None and (open_value - daily_pl) > 0:
        daily_return_pct = 100.0 * daily_pl / (open_value - daily_pl)
    unrealized = metrics.get("unrealized_pl")
    total_positions = len(active_tickers)
    cards = st.columns([1.45, 1, 1, 1])
    with cards[0]:
        st.metric("Total portfolio value", money(open_value) if open_value is not None else "—", help="Current valued ETF holdings converted to your reporting currency. Missing valuations are not treated as zero.")
    with cards[1]:
        st.metric("Today's change", money(daily_pl, signed=True) if daily_pl is not None else "—",
                  delta=(european_percent(daily_return_pct, signed=True) if daily_return_pct is not None else None),
                  help="Approximate move from available provider quote data; it may be delayed. Daily return is an estimate based on priced holdings.")
    with cards[2]:
        unrealized_pct = metrics.get("unrealized_pl_pct")
        st.metric("Unrealized profit / loss", money(unrealized, signed=True) if unrealized is not None else "—",
                  delta=(f"{european_percent(unrealized_pct, signed=True)} on cost" if unrealized_pct is not None else None),
                  help="Compared with recorded acquisition cost where cost and FX history are complete.")
    with cards[3]:
        st.metric("Investments", f"{total_positions}", help="Number of ETF tickers with a positive net quantity.")

    if not metrics.get("valuation_complete", True) and active_tickers:
        st.warning("Portfolio valuation is incomplete. One or more holdings have no usable price or FX rate; the total is withheld rather than understated.")

    # Main dashboard: performance trend and allocation, with a lightweight empty state.
    chart_col, allocation_col = st.columns([1.65, 1], gap="large")
    with chart_col:
        st.markdown('<div class="section-heading">Portfolio development</div>', unsafe_allow_html=True)
        snaps = read_snapshots(base_ccy)
        if len(snaps) >= 2:
            chart_range = st.radio("Period", ["All available", "1M", "3M", "6M", "1Y"], horizontal=True, label_visibility="collapsed", key="quick_chart_range")
            snap_view = snaps.copy()
            snap_view["as_of"] = pd.to_datetime(snap_view["as_of"], errors="coerce")
            snap_view = snap_view.dropna(subset=["as_of"])
            days = {"1M":30,"3M":90,"6M":180,"1Y":365}.get(chart_range)
            if days:
                snap_view = snap_view[snap_view["as_of"] >= (pd.Timestamp.now() - pd.Timedelta(days=days))]
            if len(snap_view) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=snap_view["as_of"], y=snap_view["market_value"], mode="lines", line=dict(color="#64748b", width=2.6), fill="tozeroy", fillcolor="rgba(100,116,139,0.11)", name="Portfolio value", hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f} " + base_ccy + "<extra></extra>"))
                fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=0,r=0), height=300, showlegend=False, xaxis=dict(showgrid=False, title=None), yaxis=dict(gridcolor="rgba(140,160,180,.12)", title=None, tickformat=",.0f"), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Not enough valuation snapshots for this period yet. Use Refresh market data to record a snapshot.")
        elif not portfolio.empty:
            st.info("Your value history will appear after at least two snapshots. Refresh market data to save a valuation snapshot.")
        else:
            st.markdown('<div class="empty-panel"><div class="empty-icon">↗</div><b>Your portfolio starts here</b><p>Add your ETFs below to see portfolio value, performance and allocation.</p></div>', unsafe_allow_html=True)
    with allocation_col:
        st.markdown('<div class="section-heading">Allocation</div>', unsafe_allow_html=True)
        valid = portfolio.dropna(subset=["market_value_base"]).copy() if not portfolio.empty else pd.DataFrame()
        if not valid.empty:
            fig = px.pie(valid, values="market_value_base", names="ticker", hole=.68, color_discrete_sequence=["#334155", "#64748b", "#94a3b8", "#cbd5e1", "#7c8798", "#b0bac7", "#475569", "#e2e8f0"])
            fig.update_traces(textposition="inside", textinfo="percent", hovertemplate="%{label}<br>%{value:,.2f} " + base_ccy + "<br>%{percent}<extra></extra>")
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=8,b=0,l=0,r=0), height=300, legend=dict(orientation="h", yanchor="top", y=-.05, x=0, font=dict(size=10)), annotations=[dict(text=f"Portfolio value<br><b>{money(open_value) if open_value is not None else '—'}</b>", x=.5, y=.5, xref="paper", yref="paper", showarrow=False, align="center", font=dict(size=12, color="#182230"))])
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.caption(f"{european_number(len(valid), decimals=0)} priced positions · {european_percent(valid['weight_pct'].max())} largest position" if "weight_pct" in valid.columns else f"{european_number(len(valid), decimals=0)} priced positions")
        else:
            st.markdown('<div class="empty-panel"><div class="empty-icon">◌</div><b>Allocation is waiting for data</b><p>Valid prices and FX rates are needed to calculate weights.</p></div>', unsafe_allow_html=True)

    holdings_head, holdings_search_col, holdings_filter_col = st.columns([1.15, 1.0, .9])
    with holdings_head:
        st.markdown('<div class="section-heading holdings-heading">Holdings <span class="small-muted">Your current positions</span></div>', unsafe_allow_html=True)
    with holdings_search_col:
        holdings_search = st.text_input("Search holdings", placeholder="Search ticker or name…", label_visibility="collapsed", key="home_holdings_search")
    with holdings_filter_col:
            holdings_filter = st.selectbox("Filter holdings", ["All holdings", "Price gainers today", "Price losers today", "Profitable", "Loss-making"], label_visibility="collapsed", key="home_holdings_filter")
    if portfolio.empty:
        st.info("No open positions yet. Use **Add investments** below to add one ETF or paste multiple rows from a spreadsheet.")
    else:
        view = portfolio.copy()
        # Friendly compact labels, preserving the underlying full-precision values.
        columns = ["ticker", "name", "quantity", "currency", "price", "day_change_pct", "price_timing", "fx_to_base", "fx_timing", "market_value_base", "unrealized_pl_base", "unrealized_pl_pct", "weight_pct", "status"]
        view = view[[c for c in columns if c in view.columns]].rename(columns={
            "ticker":"Ticker", "name":"Instrument", "quantity":"Units", "currency":"Currency", "price":"Price",
            "price_timing":"Price date / time", "fx_to_base":f"FX → {base_ccy}", "fx_timing":"FX date / time",
            "market_value_base":f"Value ({base_ccy})", "day_change_pct":"Price change %", "unrealized_pl_base":f"P/L ({base_ccy})",
            "unrealized_pl_pct":"Return %", "weight_pct":"Weight %", "status":"Status"
        })
        if holdings_search.strip():
            query = holdings_search.strip().casefold()
            mask = (view["Ticker"].astype(str).str.casefold().str.contains(query, na=False, regex=False) |
                    view["Instrument"].astype(str).str.casefold().str.contains(query, na=False, regex=False))
            view = view.loc[mask]
        if holdings_filter == "Price gainers today" and "Price change %" in view:
            view = view[pd.to_numeric(view["Price change %"], errors="coerce") > 0]
        elif holdings_filter == "Price losers today" and "Price change %" in view:
            view = view[pd.to_numeric(view["Price change %"], errors="coerce") < 0]
        elif holdings_filter == "Profitable" and f"Profit / loss ({base_ccy})" in view:
            view = view[pd.to_numeric(view[f"Profit / loss ({base_ccy})"], errors="coerce") > 0]
        elif holdings_filter == "Loss-making" and f"Profit / loss ({base_ccy})" in view:
            view = view[pd.to_numeric(view[f"Profit / loss ({base_ccy})"], errors="coerce") < 0]
        export_col, count_col = st.columns([1, 5])
        with export_col:
            st.download_button("Export filtered CSV", view.to_csv(index=False).encode("utf-8"), "investment_pilot_holdings.csv", "text/csv", use_container_width=True, disabled=view.empty)
        with count_col:
            st.caption(f"Showing {len(view)} of {len(portfolio)} positions · Values in {base_ccy} where applicable")
        if view.empty:
            st.info("No holdings match the selected search or filter.")
        else:
            european_dataframe(view, use_container_width=True, hide_index=True, height=min(470, 48 + 34 * len(view)), row_height=34, column_config={
                "Ticker":st.column_config.TextColumn(width="small"),
                "Instrument":st.column_config.TextColumn(width="medium"),
                "Units":st.column_config.NumberColumn(format="%.0f", width="small"),
                "Price":st.column_config.NumberColumn(format="%.4f", width="small"),
                "Currency":st.column_config.TextColumn(width="small"),
                "Price date / time":st.column_config.TextColumn(width="medium", help=f"Exchange price date and quote fetch time in {display_timezone}."),
                f"FX → {base_ccy}":st.column_config.NumberColumn(format="%.5f", width="small"),
                "FX date / time":st.column_config.TextColumn(width="medium", help=f"Current spot FX quote date and fetch time in {display_timezone}; a daily reference rate is used if a spot quote is unavailable."),
                f"Value ({base_ccy})":st.column_config.NumberColumn(format="%.2f", width="small"),
                "Price change %":st.column_config.NumberColumn(format="%+.2f%%", width="small", help="Today's instrument price move in its listing currency; daily portfolio P/L also includes the FX move."),
                f"P/L ({base_ccy})":st.column_config.NumberColumn(format="%+.2f", width="small"),
                "Return %":st.column_config.NumberColumn(format="%+.2f%%", width="small"),
                "Weight %":st.column_config.NumberColumn(format="%.2f%%", width="small"),
                "Status":st.column_config.TextColumn(width="small"),
            })
        if not market_data_status.empty and "status" in market_data_status.columns and (market_data_status["status"] != "OK").any():
            st.warning("Some market data is incomplete or stale. Check Data status and price dates before relying on the totals.")

    # User's main workflow stays in one place: add, adjust, sell/retract, import, or remove.
    st.markdown('<div class="section-heading action-heading">Manage your investments</div>', unsafe_allow_html=True)
    add_tab, adjust_tab, import_tab, remove_tab = st.tabs(["＋ Add / update", "↕ Adjust units", "⇧ Import CSV", "− Remove from portfolio"])
    ccy_options = ["PLN","EUR","USD","GBP","CHF","SEK","DKK","NOK","JPY","CAD","AUD"]

    with add_tab:
        st.write("Add one ETF or many at once. Existing opening-only positions can be updated here; positions with a transaction history stay intact and should be increased/decreased in **Adjust units**.")
        template = pd.DataFrame([
            {"ticker":"", "quantity":0.0, "average_cost":0.0, "currency":"EUR", "display_name":""},
            {"ticker":"", "quantity":0.0, "average_cost":0.0, "currency":"USD", "display_name":""},
        ])
        with st.form("quick_holdings_form", clear_on_submit=True):
            quick_df = st.data_editor(template, num_rows="dynamic", use_container_width=True, hide_index=True, key="quick_holdings_editor_v2", column_config={
                "ticker":st.column_config.TextColumn("Ticker / listing", help="Examples: VWCE.DE, EUNL.DE, CSPX.L"),
                "quantity":st.column_config.NumberColumn("Units", min_value=0.0, format="%.0f", width="small"),
                "average_cost":st.column_config.NumberColumn("Average cost / unit", min_value=0.0, format="localized", step=0.01, help="Cost in the currency selected on this row. Use 0 if unknown."),
                "currency":st.column_config.SelectboxColumn("Cost currency", options=ccy_options),
                "display_name":st.column_config.TextColumn("Your name (optional)"),
            })
            opening_date = st.date_input("Position date", value=date.today(), key="quick_opening_date_v2")
            quick_submit = st.form_submit_button("Save filled rows", type="primary", use_container_width=True)
        if quick_submit:
            cleaned=quick_df.copy()
            cleaned["ticker"]=cleaned["ticker"].fillna("").astype(str).str.strip().str.upper()
            cleaned["quantity"]=pd.to_numeric(cleaned["quantity"],errors="coerce").fillna(0)
            cleaned["average_cost"]=pd.to_numeric(cleaned["average_cost"],errors="coerce").fillna(0)
            rows=cleaned[(cleaned["ticker"]!="") & (cleaned["quantity"]>0)]
            successes=[]; errors=[]
            if rows.empty:
                st.warning("Enter at least one ticker and a quantity greater than zero.")
            elif rows["ticker"].duplicated().any():
                st.error("There are duplicate tickers in this batch. Combine them into one row before saving.")
            else:
                for _,row in rows.iterrows():
                    ticker=row["ticker"]
                    try:
                        status=replace_quick_position(ticker, float(row["quantity"]), float(row["average_cost"]), str(row["currency"] or "EUR").upper(), opening_date.isoformat(), str(row.get("display_name", "") or ""))
                        successes.append(f"{ticker} ({status})")
                    except ValueError as exc:
                        errors.append(f"{ticker}: {exc}")
                if successes:
                    st.session_state.quotes=None
                    st.success("Saved: " + ", ".join(successes))
                for error in errors: st.error(error)
                if successes: st.rerun()

    with adjust_tab:
        st.write("Record a real buy or sale against an existing position. This preserves your transaction history instead of silently overwriting it.")
        if not active_tickers:
            st.info("Add an ETF first to adjust its units.")
        else:
            with st.form("quick_trade_form", clear_on_submit=True):
                selected_ticker=st.selectbox("Investment", active_tickers, key="quick_trade_ticker")
                current_row=portfolio[portfolio["ticker"].astype(str).str.upper()==selected_ticker] if not portfolio.empty else pd.DataFrame()
                current_qty=float(current_row.iloc[0]["quantity"]) if not current_row.empty else 0.0
                quote_row=quotes[quotes["ticker"].astype(str).str.upper()==selected_ticker] if not quotes.empty and "ticker" in quotes else pd.DataFrame()
                quote_price=float(quote_row.iloc[0]["price"]) if not quote_row.empty and pd.notna(quote_row.iloc[0].get("price")) else 0.0
                quote_ccy=resolve_position_currency(selected_ticker, quotes, transactions, base_ccy)
                st.caption(f"Current quantity: {european_number(current_qty, decimals=0)} units · latest known quote: {european_number(quote_price)} {quote_ccy}" if quote_price else f"Current quantity: {european_number(current_qty, decimals=0)} units · no usable quote; enter execution price manually in {quote_ccy}.")
                trade_kind=st.radio("Action", ["Buy more units", "Sell / retract units"], horizontal=True, key="quick_trade_kind")
                a,b,c=st.columns(3)
                trade_qty=a.number_input("Units", min_value=0.000001, value=1.0, step=1.0, format="%.6f", key="quick_trade_qty")
                trade_price=b.number_input("Execution price / unit", min_value=0.0, value=max(quote_price,0.0), step=1.0, format="%.6f", key="quick_trade_price", help="Enter the actual transaction price for accurate cost/P&L accounting.")
                trade_fee=c.number_input("Fee", min_value=0.0, value=0.0, step=0.1, format="%.4f", key="quick_trade_fee")
                trade_date=date.today()
                trade_date=st.date_input("Trade date", value=trade_date, key="quick_trade_date")
                trade_notes=st.text_input("Notes (optional)", key="quick_trade_notes")
                trade_submit=st.form_submit_button("Record buy" if trade_kind=="Buy more units" else "Record sale", type="primary", use_container_width=True)
            if trade_submit:
                try:
                    if trade_price <= 0: raise ValueError("Enter a positive execution price.")
                    if trade_kind=="Sell / retract units" and trade_qty > current_qty + 1e-8: raise ValueError(f"You only hold {european_number(current_qty, decimals=0)} units. The sale cannot exceed the current quantity.")
                    tx_type="BUY" if trade_kind=="Buy more units" else "SELL"
                    tx_id=add_transaction({"trade_date":trade_date.isoformat(),"ticker":selected_ticker,"transaction_type":tx_type,"quantity":float(trade_qty),"unit_price":float(trade_price),"currency":quote_ccy,"amount":0.0,"fee":float(trade_fee),"notes":trade_notes or "Recorded from Investments"})
                    st.success(f"{tx_type.title()} recorded as transaction #{tx_id}.")
                    st.session_state.quotes=None
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with import_tab:
        st.write("Import multiple holdings from a spreadsheet. Review the preview and check duplicate tickers before committing.")
        st.download_button("Download CSV template", pd.DataFrame(columns=["ticker","quantity","average_cost","currency","display_name"]).to_csv(index=False).encode(), "etf_holdings_template.csv", "text/csv")
        uploaded=st.file_uploader("Choose holdings CSV", type=["csv"], key="quick_holdings_csv_v2")
        if uploaded is not None:
            try:
                import_df=pd.read_csv(uploaded); import_df.columns=import_df.columns.str.lower()
                required={"ticker","quantity","average_cost","currency"}; missing=required-set(import_df.columns)
                if missing: st.error("CSV is missing required columns: "+", ".join(sorted(missing)))
                else:
                    european_dataframe(import_df,use_container_width=True,hide_index=True)
                    import_date=st.date_input("Opening date",value=date.today(),key="csv_opening_date_v2")
                    if st.button("Import holdings",type="primary", key="quick_csv_import_v2"):
                        successes=[]; errors=[]
                        normalized=import_df.copy(); normalized["ticker"]=normalized["ticker"].fillna("").astype(str).str.strip().str.upper()
                        if normalized["ticker"].replace("",np.nan).dropna().duplicated().any():
                            st.error("The file contains duplicate tickers. Correct the CSV before importing.")
                        else:
                            for _,row in normalized.iterrows():
                                ticker=str(row.get("ticker","")).strip().upper()
                                if not ticker: continue
                                try:
                                    qty=float(row.get("quantity",0)); cost=float(row.get("average_cost",0) or 0)
                                    if qty<=0: raise ValueError("quantity must be greater than zero")
                                    status=replace_quick_position(ticker,qty,cost,str(row.get("currency","EUR") or "EUR").upper(),import_date.isoformat(),str(row.get("display_name","") or ""))
                                    successes.append(f"{ticker} ({status})")
                                except (ValueError,TypeError) as exc: errors.append(f"{ticker}: {exc}")
                            if successes:
                                st.session_state.quotes=None; st.success("Imported / updated: "+", ".join(successes))
                            for error in errors: st.error(error)
                            if successes: st.rerun()
            except Exception as exc: st.error(f"Could not read CSV: {exc}")

    with remove_tab:
        if not active_tickers:
            st.info("There are no open positions to remove.")
        else:
            st.write("Choose the correct action. A sale preserves your ledger and records proceeds; removing from tracking permanently erases all transactions for that ticker.")
            remove_symbol=st.selectbox("Investment to remove", active_tickers, key="quick_remove_symbol_v2")
            remove_mode=st.radio("Removal method", ["Sell all units (preserve history)", "Erase this ticker and all its history"], key="quick_remove_mode_v2")
            row=portfolio[portfolio["ticker"].astype(str).str.upper()==remove_symbol]
            current_qty=float(row.iloc[0]["quantity"]) if not row.empty else 0.0
            quote_row=quotes[quotes["ticker"].astype(str).str.upper()==remove_symbol] if not quotes.empty and "ticker" in quotes else pd.DataFrame()
            default_price=float(quote_row.iloc[0]["price"]) if not quote_row.empty and pd.notna(quote_row.iloc[0].get("price")) else 0.0
            if remove_mode=="Sell all units (preserve history)":
                st.caption(f"This records a SELL for the entire remaining quantity ({european_number(current_qty, decimals=0)} units).")
                sale_price=st.number_input("Sale price per unit",min_value=0.0,value=max(default_price,0.0),format="%.6f",key="quick_sell_all_price")
                sale_date=st.date_input("Sale date",value=date.today(),key="quick_sell_all_date")
                sale_fee=st.number_input("Sale fee",min_value=0.0,value=0.0,format="%.4f",key="quick_sell_all_fee")
                confirm_sale=st.checkbox(f"Confirm sale of all {european_number(current_qty, decimals=0)} {remove_symbol} units",key="quick_confirm_sell_all")
                if st.button("Record full sale",type="primary",disabled=(not confirm_sale or current_qty<=0 or sale_price<=0),key="quick_sell_all_button"):
                    try:
                        sale_currency=resolve_position_currency(remove_symbol, quotes, transactions, base_ccy)
                        tx_id=add_transaction({"trade_date":sale_date.isoformat(),"ticker":remove_symbol,"transaction_type":"SELL","quantity":current_qty,"unit_price":sale_price,"currency":sale_currency,"amount":0.0,"fee":sale_fee,"notes":"Full position sale recorded from Investments"})
                        st.success(f"Full sale recorded as transaction #{tx_id}; the ticker and history remain in the ledger.")
                        st.session_state.quotes=None; st.rerun()
                    except ValueError as exc: st.error(str(exc))
            else:
                st.warning("This permanently deletes every transaction for the selected ticker. It is appropriate for correcting a mistaken entry, not for recording an actual sale.")
                confirm_remove=st.checkbox(f"I understand this erases all {remove_symbol} transaction history",key="quick_confirm_erase")
                if st.button("Erase ticker history",type="secondary",disabled=not confirm_remove,key="quick_erase_button"):
                    try:
                        deleted=remove_ticker_from_portfolio(remove_symbol)
                        st.session_state.quotes=None
                        st.success(f"Removed {remove_symbol} and {deleted} transaction row(s). Instrument metadata and watchlist entry are retained.")
                        st.rerun()
                    except ValueError as exc: st.error(str(exc))

    st.caption("Investments is your portfolio home. Changes to opening-only positions update the snapshot; buys and sales are recorded as transactions. Prices and FX may be delayed or unavailable, and no trades are executed by this app.")

elif page == "Transactions":
    st.title("Transaction ledger")
    st.write("Record every buy, sale, opening position, dividend, fee and cash movement. The ledger—not a manually edited holdings table—is the source of truth for quantities and average-cost accounting.")
    # Keep the type selector outside the form so the form fields respond immediately to type changes.
    kind = st.selectbox("Transaction type", ["OPENING", "BUY", "SELL", "DIVIDEND", "FEE", "CASH_IN", "CASH_OUT"])
    with st.form("transaction_form", clear_on_submit=True):
        trade_date = st.date_input("Effective date", value=date.today())
        asset_event = kind in {"OPENING", "BUY", "SELL"}
        income_event = kind == "DIVIDEND"
        ticker = st.text_input("Ticker / listing symbol", placeholder="VWCE.DE").strip().upper() if asset_event or income_event else ""
        ccy_options = ["PLN","EUR","USD","GBP","CHF","SEK","DKK","NOK","JPY","CAD","AUD"]
        currency = st.selectbox("Transaction / cash currency", ccy_options, index=0)
        if asset_event:
            a,b,c = st.columns(3)
            quantity = a.number_input("Units", min_value=0.0, value=1.0, step=1.0, format="%.6f")
            unit_price = b.number_input("Price per unit", min_value=0.0, value=0.0, step=1.0, format="%.6f", help="For OPENING, use the average acquisition cost per unit. Enter 0 if unknown.")
            fee = c.number_input("Fee", min_value=0.0, value=0.0, step=0.1, format="%.4f")
            amount = 0.0
        else:
            amount = st.number_input("Amount received / paid", min_value=0.0, value=0.0, step=10.0, format="%.4f",
                                     help="For DIVIDEND, enter the net amount received. For FEE, enter the fee amount.")
            fee = 0.0
            quantity, unit_price = 0.0, 0.0
        notes = st.text_input("Notes (optional)", placeholder="Broker statement reference, dividend period, etc.")
        saved = st.form_submit_button("Save transaction", type="primary")
        if saved:
            try:
                tx_id = add_transaction({"trade_date":trade_date.isoformat(), "ticker":ticker, "transaction_type":kind,
                    "quantity":quantity, "unit_price":unit_price, "currency":currency, "amount":amount, "fee":fee, "notes":notes})
                st.success(f"Saved transaction #{tx_id}.")
                st.session_state.quotes = None
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.markdown("---")
    st.subheader("Recorded transactions")
    if transactions.empty:
        st.info("No transactions yet.")
    else:
        tx_view = transactions.sort_values(["trade_date","id"], ascending=[False,False])[[
            "id", "trade_date", "transaction_type", "ticker", "quantity", "currency", "unit_price", "amount", "fee", "notes"
        ]].rename(columns={
            "id":"#", "trade_date":"Date", "transaction_type":"Type", "ticker":"Ticker", "quantity":"Units",
            "currency":"Currency", "unit_price":"Unit price", "amount":"Amount", "fee":"Fee", "notes":"Notes",
        })
        tx_view["Date"] = pd.to_datetime(tx_view["Date"], errors="coerce").dt.date
        european_dataframe(tx_view, use_container_width=True, hide_index=True, row_height=34, column_config={
            "#":st.column_config.NumberColumn(width="small", format="%d"),
            "Date":st.column_config.DateColumn(width="small"),
            "Type":st.column_config.TextColumn(width="small"),
            "Ticker":st.column_config.TextColumn(width="small"),
            "Units":st.column_config.NumberColumn(width="small", format="%.0f"),
            "Unit price":st.column_config.NumberColumn(width="small", format="%.4f"),
            "Currency":st.column_config.TextColumn(width="small"),
            "Amount":st.column_config.NumberColumn(width="small", format="%.2f"),
            "Fee":st.column_config.NumberColumn(width="small", format="%.2f"),
            "Notes":st.column_config.TextColumn(width="medium"),
        })
        st.download_button("Download ledger CSV", transactions.to_csv(index=False).encode("utf-8"), "etf_transactions.csv", "text/csv")
        ids = transactions["id"].astype(int).tolist()
        with st.form("delete_transaction_form"):
            selected_id = st.selectbox("Delete a transaction (recalculates the ledger)", ids, format_func=lambda x: f"#{x} · {transactions.loc[transactions.id == x, 'trade_date'].iloc[0]} · {transactions.loc[transactions.id == x, 'transaction_type'].iloc[0]} · {transactions.loc[transactions.id == x, 'ticker'].iloc[0]}")
            confirm_delete = st.checkbox("I understand this changes quantities and historical P/L")
            delete_pressed = st.form_submit_button("Delete selected transaction")
            if delete_pressed:
                if not confirm_delete:
                    st.error("Confirm the deletion first.")
                else:
                    try:
                        delete_transaction(int(selected_id))
                        st.session_state.quotes = None
                        st.success("Transaction deleted.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

elif page == "Performance":
    st.title("Performance, benchmark & contribution attribution")
    st.markdown('<div class="terminal-tag">ADJUSTED-CLOSE TOTAL RETURN · FX CONVERTED · SOURCE-AWARE</div>', unsafe_allow_html=True)
    st.write("Compare an ETF-holdings total-return index with a benchmark on common dates and in the selected reporting currency. This is not the same as brokerage-account return: cash balances, taxes and external cash yield are excluded.")
    if portfolio.empty or not active_tickers:
        st.info("Add holdings first to build a historical performance series.")
    else:
        benchmark_options = {
            "S&P 500 Index (^GSPC)": "^GSPC",
            "Euro STOXX 50 (^STOXX50E)": "^STOXX50E",
            "NASDAQ Composite (^IXIC)": "^IXIC",
            "MSCI World ETF proxy (IWDA.AS)": "IWDA.AS",
            "Custom Yahoo Finance symbol": "",
        }
        with st.form("historical_performance_controls"):
            form_a, form_b, form_c = st.columns([1, 1.5, 1.5])
            period_choice = form_a.selectbox("History window", ["1y", "3y", "5y"], index=1, key="perf_history_window")
            benchmark_label = form_b.selectbox("Benchmark", list(benchmark_options.keys()), index=0, key="perf_benchmark_choice")
            custom_benchmark = form_c.text_input("Custom benchmark symbol", value=st.session_state.get("perf_custom_symbol", "IWDA.AS"), key="perf_custom_symbol_input", help="Used when Custom Yahoo Finance symbol is selected.")
            build_performance = st.form_submit_button("Apply settings & refresh history", type="primary")
        benchmark_symbol = benchmark_options[benchmark_label] or custom_benchmark.strip().upper()
        if not benchmark_symbol:
            st.warning("Enter a benchmark ticker supported by Yahoo Finance.")
        else:
            with st.spinner("Loading adjusted history and historical currency rates…"):
                symbols = sorted(set(active_tickers + [benchmark_symbol]))
                local_history = get_history(symbols, period=period_choice, force_refresh=build_performance)
                benchmark_quote = get_quotes([benchmark_symbol], force_refresh=build_performance)
                all_quotes = pd.concat([quotes, benchmark_quote], ignore_index=True) if not quotes.empty else benchmark_quote.copy()
                if not all_quotes.empty:
                    all_quotes = all_quotes.drop_duplicates("ticker", keep="last")
                currency_by_ticker = {str(r["ticker"]).upper(): str(r.get("currency", "") or "").upper()
                                      for _, r in all_quotes.iterrows()}
                price_start = local_history.index.min().date().isoformat() if not local_history.empty else date.today().isoformat()
                price_end = local_history.index.max().date().isoformat() if not local_history.empty else date.today().isoformat()
                fx_history_by_currency = {}
                required_currencies = {currency_by_ticker.get(t, "") for t in symbols}
                for currency in sorted(required_currencies - {"", base_ccy}):
                    fx_history_by_currency[currency] = get_fx_history(
                        currency, base_ccy, price_start, price_end, force_refresh=build_performance
                    )
                converted_prices = convert_prices_to_currency(local_history, currency_by_ticker, fx_history_by_currency, base_ccy)
                holding_columns = [t for t in active_tickers if t in converted_prices.columns]
                if holding_columns:
                    portfolio_prices = converted_prices[holding_columns]
                    performance = portfolio_total_return(portfolio_prices, transactions)
                else:
                    performance = pd.DataFrame()
                benchmark_prices = converted_prices[benchmark_symbol] if benchmark_symbol in converted_prices.columns else pd.Series(dtype=float)
                comparison = compare_to_benchmark(performance, benchmark_prices)
            missing_currency_symbols = [symbol for symbol in symbols if not currency_by_ticker.get(symbol)]
            if local_history.empty:
                st.error("No historical price series was returned. Check ticker symbols and the Market Data status below.")
            elif missing_currency_symbols:
                st.warning("Listing currency metadata is missing for: " + ", ".join(missing_currency_symbols) + ". Currency-adjusted total return cannot be trusted until the listing currency is identified.")
            elif comparison.empty:
                st.warning("There are not enough common price/FX observations to compare the portfolio and benchmark. Check historical data coverage and FX availability.")
            else:
                common_perf = performance.reindex(comparison.index)
                p_total = float((1 + comparison["Portfolio return"]).prod() - 1)
                b_total = float((1 + comparison["Benchmark return"]).prod() - 1)
                elapsed_days = max((comparison.index[-1] - comparison.index[0]).days, 1)
                p_ann = (1 + p_total) ** (365.25 / elapsed_days) - 1 if p_total > -1 else np.nan
                b_ann = (1 + b_total) ** (365.25 / elapsed_days) - 1 if b_total > -1 else np.nan
                summary = performance_summary(common_perf)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Portfolio total return", european_percent(p_total * 100, signed=True))
                m2.metric(f"{benchmark_symbol} total return", european_percent(b_total * 100, signed=True))
                m3.metric("Relative return", european_percent((p_total - b_total) * 100, signed=True), help="Portfolio cumulative return minus benchmark cumulative return; not a risk-adjusted alpha estimate.")
                m4.metric("Portfolio annualized return", european_percent(p_ann * 100, signed=True) if np.isfinite(p_ann) else "—")
                m5.metric("Max drawdown", european_percent(summary["max_drawdown"] * 100, signed=True) if summary.get("max_drawdown") is not None else "—")
                chart = comparison[["Portfolio index", "Benchmark index"]].reset_index().melt(id_vars="date", var_name="Series", value_name="Index (start = 100)")
                fig = px.line(chart, x="date", y="Index (start = 100)", color="Series", title=f"Historical total-return comparison in {base_ccy} · benchmark {benchmark_symbol}")
                fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Annualized volatility", european_percent(summary["annualized_volatility"] * 100) if summary.get("annualized_volatility") is not None else "—")
                c2.metric("Valid daily observations", european_number(summary.get("observations", 0), decimals=0))
                c3.metric("Benchmark annualized return", european_percent(b_ann * 100, signed=True) if np.isfinite(b_ann) else "—")
                st.subheader("Linked ticker contribution to portfolio return")
                attribution = linked_attribution(common_perf)
                if not attribution.empty:
                    european_dataframe(attribution, use_container_width=True, hide_index=True,
                                 column_config={"Contribution %": st.column_config.NumberColumn(format="%+.3f%%")})
                    fig2 = px.bar(attribution.sort_values("Contribution %"), x="Contribution %", y="Ticker", orientation="h",
                                  title="Geometrically linked contributions over the displayed period")
                    fig2.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig2, use_container_width=True)
                    total_contribution = float(attribution["Contribution %"].sum())
                    st.caption(f"Linked contributions sum to {european_percent(total_contribution, signed=True)} versus portfolio return {european_percent(p_total*100, signed=True)}. Small rounding differences may occur.")
                else:
                    st.info("No complete asset-contribution series was available for this period.")
                st.caption("Method: provider-adjusted close series (dividend/split-adjusted), daily dated Frankfurter/ECB reference FX rates, daily holdings from the transaction ledger, and a Carino-style geometric linking of daily ticker contributions. ETF holdings are assumed to reinvest fund distributions as represented by adjusted prices. Cash balances, taxes, withholding, exact trade-time effects and non-ETF assets are excluded. Yahoo/FX gaps can affect the result; this is research analytics, not a broker statement.")
        st.divider()
        value = metrics.get("value")
        a, b, c, d = st.columns(4)
        a.metric("Current ETF value", money(value))
        a_value = portfolio_xirr(transactions, float(value), date.today().isoformat(), base_ccy, historical_fx_for_xirr) if metrics.get("valuation_complete") and value is not None else None
        b.metric("Estimated money-weighted XIRR", european_percent(a_value * 100, signed=True) if a_value is not None else "—", help="Approximate cash-flow-aware return using recorded investment flows and current ETF value. Excludes cash balances, taxes and non-ETF assets.")
        c.metric("Recorded dividends", money(analytics.get("dividends", 0)))
        d.metric("Recorded fees", money(analytics.get("fees", 0)))
        st.subheader("Recorded valuation snapshots (not a return index)")
        snaps = read_snapshots(base_ccy)
        if len(snaps) >= 2:
            snaps = snaps.sort_values("as_of")
            fig = px.line(snaps, x="as_of", y="market_value", markers=True, labels={"as_of": "Snapshot", "market_value": f"ETF market value ({base_ccy})"})
            fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Snapshots are point-in-time market values and are not adjusted for contributions or historical portfolio composition.")
        else:
            st.caption("Scheduled or manual refreshes create valuation snapshots when the current valuation is complete.")
        st.subheader("Investment flows used by XIRR")
        cf = transactions[transactions["transaction_type"].isin(["OPENING", "BUY", "SELL", "DIVIDEND"])].copy() if not transactions.empty else pd.DataFrame()
        if cf.empty:
            st.caption("No investment transaction entries recorded yet.")
        else:
            european_dataframe(cf[[c for c in ["trade_date", "transaction_type", "amount", "currency", "ticker", "notes"] if c in cf.columns]], use_container_width=True, hide_index=True)

elif page == "ETF Research":
    st.title("ETF research & instrument master")
    st.markdown('<div class="terminal-tag">INSTRUMENT IDENTITY · FUND NOTES · OVERLAP READINESS</div>', unsafe_allow_html=True)
    st.write("Maintain stable identifiers and your own display labels. Automated country/sector look-through requires a trusted holdings dataset; the app will not invent fund constituents.")
    with st.form("instrument_master_form"):
        ticker_input = st.text_input("Ticker / listing symbol", placeholder="VWCE.DE").strip().upper()
        name_input = st.text_input("Preferred display name", placeholder="Vanguard FTSE All-World UCITS ETF")
        i1, i2 = st.columns(2)
        isin_input = i1.text_input("ISIN (optional)", max_chars=32)
        exchange_input = i2.text_input("Exchange (optional)", placeholder="XETRA")
        asset_class_input = st.selectbox("Asset class", ["ETF", "Equity", "Bond ETF", "Commodity ETF", "Money market", "Other"])
        instrument_notes = st.text_area("Research notes", height=80)
        save_instrument_button = st.form_submit_button("Save instrument", type="primary")
        if save_instrument_button:
            try:
                save_instrument(ticker_input, name_input, isin_input, exchange_input, asset_class_input, instrument_notes)
                st.success(f"Saved instrument metadata for {ticker_input}.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    instruments = list_instruments()
    if not instruments.empty:
        st.subheader("Instrument master")
        european_dataframe(instruments, use_container_width=True, hide_index=True)
        st.download_button("Export instrument master CSV", instruments.to_csv(index=False).encode("utf-8"), "etf_instruments.csv", "text/csv")
    st.subheader("Portfolio exposure overview")
    if not portfolio.empty and portfolio["market_value_base"].notna().any():
        priced = portfolio.dropna(subset=["market_value_base"]).copy()
        priced["value"] = priced["market_value_base"]
        fig = px.treemap(priced, path=["ticker"], values="value", color="weight_pct", color_continuous_scale="Greys", title=f"Priced positions by market value ({base_ccy})")
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This is exposure by held ETF, not look-through exposure to underlying securities. Underlying holdings overlap and sector/geography are not inferred without source data.")
    else:
        st.info("Exposure visualization will appear after at least one position has a valid price and FX rate.")

    st.divider()
    st.subheader("Sourced ETF constituent data & overlap")
    st.write("Import issuer/factsheet holdings as of a stated date. The app never invents constituents. Overlap is calculated only from imported rows and can be understated when your source includes only top holdings.")
    template = "fund_ticker,constituent_ticker,weight_pct,constituent_name\n".encode("utf-8")
    st.download_button("Download CSV template", template, "etf_constituents_template.csv", "text/csv")
    with st.expander("Import a holdings CSV", expanded=False):
        source_name = st.text_input("Data source / issuer factsheet name", placeholder="ETF issuer factsheet, holdings export or licensed provider", key="holdings_source_name")
        source_url = st.text_input("Source URL (optional for local files)", placeholder="https://…", key="holdings_source_url")
        as_of_date = st.date_input("Holdings as-of date", value=date.today(), key="holdings_as_of_date")
        holdings_upload = st.file_uploader("CSV holdings file", type=["csv"], key="etf_holdings_upload")
        parsed_holdings = None
        if holdings_upload is not None:
            try:
                parsed_holdings = pd.read_csv(holdings_upload)
                st.caption(f"Preview · {european_number(len(parsed_holdings), decimals=0)} rows")
                european_dataframe(parsed_holdings.head(20), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
        if st.button("Import sourced constituent snapshot", type="primary", disabled=parsed_holdings is None, key="import_etf_holdings_button"):
            required_columns = {"fund_ticker", "constituent_ticker", "weight_pct"}
            missing_columns = required_columns - set(parsed_holdings.columns) if parsed_holdings is not None else required_columns
            if missing_columns:
                st.error("CSV is missing required column(s): " + ", ".join(sorted(missing_columns)))
            elif not source_name.strip():
                st.error("Provide the factsheet/provider name so the imported data has provenance.")
            else:
                candidate = parsed_holdings.copy()
                candidate["fund_ticker"] = candidate["fund_ticker"].fillna("").astype(str).str.strip().str.upper()
                candidate["constituent_ticker"] = candidate["constituent_ticker"].fillna("").astype(str).str.strip().str.upper()
                candidate["weight_pct"] = pd.to_numeric(candidate["weight_pct"], errors="coerce")
                has_name = "constituent_name" in candidate.columns
                if candidate["fund_ticker"].eq("").any() or candidate["constituent_ticker"].eq("").any():
                    st.error("Each row needs both a fund_ticker and constituent_ticker.")
                elif candidate["weight_pct"].isna().any() or (candidate["weight_pct"] < 0).any() or (candidate["weight_pct"] > 100).any():
                    st.error("Weights must be numeric percentage points between 0 and 100 (for example 4.25 means 4.25%).")
                elif candidate.duplicated(["fund_ticker", "constituent_ticker"]).any():
                    st.error("Duplicate fund_ticker + constituent_ticker rows found. Clean duplicates before import.")
                else:
                    try:
                        counts = {}
                        for fund, group in candidate.groupby("fund_ticker"):
                            selected_cols = ["constituent_ticker", "weight_pct"] + (["constituent_name"] if has_name else [])
                            counts[fund] = save_etf_holdings(fund, group[selected_cols], source_name=source_name,
                                source_url=source_url, as_of=as_of_date.isoformat())
                        st.success("Imported sourced holdings for " + ", ".join(f"{fund} ({n} rows)" for fund, n in counts.items()))
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
    latest_holdings = list_etf_holdings(latest_only=True)
    if latest_holdings.empty:
        st.info("No constituent snapshots imported yet. Download the CSV template, populate it from an issuer factsheet/provider export, and record its source and as-of date.")
    else:
        st.subheader("Imported dataset coverage & provenance")
        coverage = holdings_weight_coverage(latest_holdings)
        european_dataframe(coverage, use_container_width=True, hide_index=True)
        european_dataframe(latest_holdings[[c for c in ["fund_ticker", "constituent_ticker", "constituent_name", "weight_pct", "source_name", "source_url", "as_of", "imported_at"] if c in latest_holdings.columns]], use_container_width=True, hide_index=True)
        available_funds = sorted(latest_holdings["fund_ticker"].unique().tolist())
        held_defaults = [fund for fund in available_funds if fund in active_tickers]
        default_funds = held_defaults if len(held_defaults) >= 2 else available_funds
        selected_funds = st.multiselect("Funds to compare", available_funds, default=default_funds, key="overlap_funds_selection")
        selected_holdings = latest_holdings[latest_holdings["fund_ticker"].isin(selected_funds)].copy()
        selected_coverage = holdings_weight_coverage(selected_holdings)
        if len(selected_funds) >= 2:
            overlap_summary, overlap_details = calculate_etf_overlap(selected_holdings)
            if not selected_coverage.empty and (selected_coverage["weight_total_pct"] < 90).any():
                st.warning("At least one source snapshot covers less than 90% of total fund weight. The overlap estimate may be materially understated; compare only like-for-like complete holdings snapshots when possible.")
            if not selected_coverage.empty and selected_coverage["as_of"].nunique() > 1:
                st.warning("The selected fund holdings are from different as-of dates. Overlap can change over time; compare synchronized snapshots for a cleaner result.")
            if not overlap_summary.empty:
                st.subheader("Pairwise shared-weight overlap")
                european_dataframe(overlap_summary, use_container_width=True, hide_index=True,
                             column_config={"Overlap %": st.column_config.NumberColumn(format="%.2f%%")})
                st.caption("Overlap formula: sum of the smaller reported weight for each shared constituent. This is not sector/country exposure or a forecast. Snapshot dates and source quality differ by fund.")
                if not overlap_details.empty:
                    with st.expander("Show shared constituents behind the score"):
                        european_dataframe(overlap_details, use_container_width=True, hide_index=True,
                                     column_config={"Weight A %": st.column_config.NumberColumn(format="%.3f%%"),
                                                    "Weight B %": st.column_config.NumberColumn(format="%.3f%%"),
                                                    "Overlap contribution %": st.column_config.NumberColumn(format="%.3f%%")})
            else:
                st.info("No shared constituents were found in the supplied snapshots.")
        else:
            st.info("Select at least two funds with imported constituent data to calculate overlap.")

elif page == "Watchlist":
    st.title("Instrument watchlist")
    st.markdown('<div class="terminal-tag">TRACK GLOBAL MARKETS · FX · CRYPTO · BONDS · COMMODITIES · FUNDS</div>', unsafe_allow_html=True)
    st.write("Watch almost any market instrument supported by Yahoo Finance, whether or not you own it. Use a provider symbol for the instrument; prices, currencies and availability depend on the data source.")

    # Curated Yahoo Finance symbols make non-equity instruments discoverable without
    # changing the existing ticker-keyed database or quote/history pipeline.
    watchlist_presets = {
        "FX · EUR/PLN spot (EURPLN=X)": ("EURPLN=X", "FX · EUR/PLN exchange rate"),
        "FX · EUR/USD spot (EURUSD=X)": ("EURUSD=X", "FX · EUR/USD exchange rate"),
        "FX · GBP/USD spot (GBPUSD=X)": ("GBPUSD=X", "FX · GBP/USD exchange rate"),
        "FX · USD/JPY spot (JPY=X)": ("JPY=X", "FX · USD/JPY exchange rate"),
        "Crypto · Bitcoin (BTC-USD)": ("BTC-USD", "Crypto · Bitcoin in USD"),
        "Crypto · Ethereum (ETH-USD)": ("ETH-USD", "Crypto · Ethereum in USD"),
        "Commodity · Gold futures (GC=F)": ("GC=F", "Commodity · Gold futures"),
        "Commodity · Silver futures (SI=F)": ("SI=F", "Commodity · Silver futures"),
        "Commodity · WTI crude oil futures (CL=F)": ("CL=F", "Commodity · WTI crude oil futures"),
        "Commodity · Natural gas futures (NG=F)": ("NG=F", "Commodity · Natural gas futures"),
        "Commodity · Copper futures (HG=F)": ("HG=F", "Commodity · Copper futures"),
        "Rates · US 10-year Treasury yield (^TNX)": ("^TNX", "Rates · US 10-year Treasury yield index, not bond price"),
        "Bonds · US 10-year Treasury note futures (ZN=F)": ("ZN=F", "Bonds · US 10-year Treasury note futures"),
        "Bonds · US 20+ year Treasury ETF (TLT)": ("TLT", "Bond ETF · Long-duration US Treasuries"),
        "Bonds · US 7–10 year Treasury ETF (IEF)": ("IEF", "Bond ETF · Intermediate US Treasuries"),
        "Index · S&P 500 (^GSPC)": ("^GSPC", "Index · S&P 500"),
        "Index · Nasdaq Composite (^IXIC)": ("^IXIC", "Index · Nasdaq Composite"),
        "Index · Euro Stoxx 50 (^STOXX50E)": ("^STOXX50E", "Index · Euro Stoxx 50"),
        "Volatility · VIX (^VIX)": ("^VIX", "Volatility · CBOE VIX index"),
    }
    quick_col, quick_button_col = st.columns([3, 1])
    with quick_col:
        preset_label = st.selectbox("Quick-add a popular instrument", list(watchlist_presets), key="watchlist_preset")
    with quick_button_col:
        st.write("")
        st.write("")
        if st.button("Add preset", type="primary", use_container_width=True):
            preset_symbol, preset_note = watchlist_presets[preset_label]
            try:
                add_watchlist_ticker(preset_symbol, preset_note)
                st.success(f"Added {preset_symbol} to your watchlist.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with st.expander("Add any other instrument by market symbol", expanded=True):
        st.caption("Examples: EURPLN=X, EURUSD=X, BTC-USD, ETH-USD, GC=F (gold futures), ^TNX (US 10-year yield), ZN=F (10-year Treasury futures), TLT, CSPX.L. FX pairs commonly use =X, futures often use =F, and indices often begin with ^ on Yahoo Finance.")
        with st.form("watchlist_add_form"):
            w1, w2 = st.columns([1, 2])
            watch_ticker = w1.text_input("Market symbol", placeholder="EURPLN=X or BTC-USD").strip().upper()
            watch_note = w2.text_input("Description / note (optional)", placeholder="FX pair, commodity, bond, index…")
            watch_add = st.form_submit_button("Add instrument", type="primary")
            if watch_add:
                try:
                    add_watchlist_ticker(watch_ticker, watch_note)
                    st.success(f"Added {watch_ticker} to watchlist.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    watch = list_watchlist()
    if not watch.empty:
        watch_quotes = add_quote_timing_columns(get_quotes(watch["ticker"].tolist()))
        watch_view = watch.merge(watch_quotes, on="ticker", how="left")
        st.subheader(f"Your instruments · {len(watch_view)}")
        watch_view.insert(0, "order", range(1, len(watch_view) + 1))
        watch_table = watch_view[[c for c in ["order", "ticker", "name", "currency", "price", "day_change_pct", "price_timing", "status", "note"] if c in watch_view.columns]].rename(columns={
            "order":"Order", "ticker":"Symbol", "name":"Name", "currency":"Currency", "price":"Price", "day_change_pct":"Price change %",
            "price_timing":"Price date / time", "status":"Status", "note":"Note",
        })
        european_dataframe(watch_table, use_container_width=True, hide_index=True, row_height=34, column_config={
            "Order":st.column_config.NumberColumn(format="%.0f", width="small"),
            "Symbol":st.column_config.TextColumn(width="small"),
            "Name":st.column_config.TextColumn(width="medium"),
            "Currency":st.column_config.TextColumn(width="small"),
            "Price":st.column_config.NumberColumn(format="%.4f", width="small"),
            "Price change %":st.column_config.NumberColumn(format="%+.2f%%", width="small"),
            "Price date / time":st.column_config.TextColumn(width="medium", help=f"Exchange price date and quote fetch time in {display_timezone}."),
            "Status":st.column_config.TextColumn(width="small"),
            "Note":st.column_config.TextColumn(width="medium"),
        })
        st.download_button("Export watchlist CSV", watch_view.to_csv(index=False).encode("utf-8"), "investment_pilot_watchlist.csv", "text/csv")
        with st.expander("Change instrument order"):
            st.caption("Edit the Order numbers, then save. Use a different whole number for each instrument; 1 appears first.")
            order_editor_data = watch[["ticker", "sort_order"]].copy()
            order_editor_data["sort_order"] = pd.to_numeric(order_editor_data["sort_order"], errors="coerce").fillna(0).astype(int)
            order_editor_data = order_editor_data.rename(columns={"ticker":"Symbol", "sort_order":"Order"})
            order_key = hashlib.sha1("|".join(watch["ticker"].astype(str)).encode("utf-8")).hexdigest()[:10]
            edited_order = st.data_editor(
                order_editor_data, use_container_width=True, hide_index=True, key=f"watchlist_order_editor_{order_key}",
                column_config={
                    "Symbol":st.column_config.TextColumn(disabled=True, width="medium"),
                    "Order":st.column_config.NumberColumn(min_value=1, step=1, format="%d", width="small"),
                },
                disabled=["Symbol"],
            )
            if st.button("Save instrument order", type="primary", key="save_watchlist_order"):
                order_values = pd.to_numeric(edited_order["Order"], errors="coerce")
                if (order_values.isna().any() or order_values.le(0).any()
                        or order_values.mod(1).ne(0).any() or order_values.nunique() != len(order_values)):
                    st.error("Give every instrument a different whole-number order.")
                else:
                    ordered_tickers = edited_order.assign(_order=order_values).sort_values("_order")["Symbol"].tolist()
                    try:
                        save_watchlist_order(ordered_tickers)
                        st.success("Your instrument order was saved.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        remove_symbol = st.selectbox("Remove symbol", watch["ticker"].tolist())
        if st.button("Remove selected symbol"):
            remove_watchlist_ticker(remove_symbol)
            st.rerun()
    else:
        st.info("Your watchlist is empty. Add an FX pair, cryptocurrency, bond/rate instrument, commodity, index, ETF or stock using the quick-add menu or a provider symbol.")

elif page == "Holdings & risk":
    st.title("Holdings & risk")
    if portfolio.empty:
        st.info("Record OPENING positions or BUY transactions first.")
    else:
        display_position_table(portfolio)
        valid = portfolio.dropna(subset=["weight_pct"])
        if not valid.empty:
            hhi = float((valid["weight_pct"] ** 2).sum())
            c1,c2,c3 = st.columns(3)
            c1.metric("Concentration index (HHI)", f"{european_number(hhi, decimals=0)} / 10 000")
            c2.metric("Largest position", european_percent(valid["weight_pct"].max()))
            c3.metric("Quote currencies", str(portfolio["currency"].replace("", np.nan).nunique()))
            if valid["weight_pct"].max() >= 50:
                st.warning("One ETF is at least 50% of priced holdings by market value. Check whether this concentration is intentional.")
        st.subheader("Historical risk indicators")
        period = st.selectbox("History window", ["6mo","1y","3y","5y"], index=1, key="risk_period")
        if st.button("Calculate return and drawdown metrics"):
            history = get_history(active_tickers, period, force_refresh=True)
            if history.empty:
                st.warning("No usable history returned. Some tickers may not be supported by the provider.")
            else:
                returns = history.pct_change(fill_method=None).dropna(how="all")
                summary_df = historical_risk_table(history)
                if not summary_df.empty:
                    european_dataframe(summary_df, use_container_width=True, hide_index=True,
                        column_config={c:st.column_config.NumberColumn(format="%+.2f%%") for c in ["Price return %","Annualized volatility %","Maximum drawdown %","Worst daily move %"]})
                if len(returns.columns) >= 2:
                    st.subheader("Daily-return correlation")
                    corr = returns.corr(min_periods=20)
                    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
                    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                st.caption("Historical return shown is adjusted ETF price history from the data provider; verify dividend treatment and listing coverage. This is not a holdings-weighted portfolio backtest.")
        st.subheader("Risk interpretation")
        st.markdown("- **Volatility** measures dispersion of daily price returns; it is not the probability of losing money.\n- **Maximum drawdown** is the largest peak-to-trough fall in the selected history.\n- **Correlation** measures how similarly ETF prices moved in this sample; it can change materially in a crisis.\n- Trading currency is not always the same as underlying economic currency exposure. Hedging and fund domicile matter.")

elif page == "Market analysis":
    st.title("Markets & macro dashboard")
    st.write("A watchlist of common public market symbols. Coverage, timestamps and data availability depend on Yahoo Finance and the listing.")
    market_watch = {
        "S&P 500":"^GSPC", "Nasdaq Composite":"^IXIC", "Euro Stoxx 50":"^STOXX50E",
        "VIX":"^VIX", "US 10Y yield proxy":"^TNX", "EUR/PLN":"EURPLN=X", "USD/PLN":"USDPLN=X",
        "Gold futures":"GC=F", "WTI crude futures":"CL=F"
    }
    selected_names = st.multiselect("Indicators", list(market_watch.keys()), default=["S&P 500","Nasdaq Composite","EUR/PLN","USD/PLN","VIX"])
    history_window = st.selectbox("Lookback", ["1mo","3mo","6mo","1y","5y"], index=2)
    if st.button("Load market watch", type="primary"):
        symbols = [market_watch[n] for n in selected_names]
        hist = get_history(symbols, history_window, force_refresh=True)
        market_quotes = get_quotes(symbols, force_refresh=True)
        st.session_state.market_history = hist
        st.session_state.market_watch_quotes = market_quotes
        st.session_state.market_watch_selection = selected_names
        st.session_state.market_watch_symbols = symbols
    hist = st.session_state.get("market_history", pd.DataFrame())
    if not hist.empty:
        market_quotes = st.session_state.get("market_watch_quotes", pd.DataFrame())
        quote_by_symbol = (
            {str(row["ticker"]).upper():row for _, row in market_quotes.iterrows()}
            if not market_quotes.empty and "ticker" in market_quotes else {}
        )
        history_fetch_times = get_history_fetch_times(hist.columns.tolist())
        fig = go.Figure()
        for symbol in hist.columns:
            s = hist[symbol].dropna()
            if len(s):
                fig.add_trace(go.Scatter(x=s.index, y=(s/s.iloc[0]-1)*100, name=symbol, mode="lines"))
        fig.update_layout(title="Indexed price movement (period start = 0%)", yaxis_title="Change (%)", hovermode="x unified",
                          template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        returns = hist.pct_change(fill_method=None).dropna(how="all")
        rows = []
        for symbol in hist.columns:
            s = hist[symbol].dropna()
            r = s.pct_change(fill_method=None).dropna()
            if len(s) and len(r):
                quote = quote_by_symbol.get(str(symbol).upper())
                quoted_price = pd.to_numeric(quote.get("price"), errors="coerce") if quote is not None else np.nan
                latest_price = float(quoted_price) if pd.notna(quoted_price) and np.isfinite(quoted_price) else float(s.iloc[-1])
                rows.append({"Symbol":symbol,"Currency":str(quote.get("currency", "") or "") if quote is not None else "",
                             "Price":latest_price,
                             "Price change %":pd.to_numeric(quote.get("day_change_pct"), errors="coerce") if quote is not None else np.nan,
                             "Period change %":(s.iloc[-1]/s.iloc[0]-1)*100,
                             "Volatility %":r.std()*np.sqrt(252)*100,
                             "Latest / fetched":market_timing_label(str(s.index[-1])[:10], local_market_time(history_fetch_times.get(symbol, "")))})
        european_dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, row_height=34, column_config={
            "Symbol":st.column_config.TextColumn(width="small"),
            "Currency":st.column_config.TextColumn(width="small"),
            "Price":st.column_config.NumberColumn(format="%.2f", width="small"),
            "Price change %":st.column_config.NumberColumn(format="%+.2f%%", width="small"),
            "Period change %":st.column_config.NumberColumn(format="%+.2f%%", width="small"),
            "Volatility %":st.column_config.NumberColumn(format="%.2f%%", width="small"),
            "Latest / fetched":st.column_config.TextColumn(width="medium", help=f"Latest daily observation date, then history cache fetch time in {display_timezone}."),
        })
    st.subheader("Market moves that may affect ETF portfolios")
    st.markdown("**Rates:** changing yield expectations can affect bond prices and equity valuations. **FX:** a weaker foreign currency can reduce unhedged foreign-asset returns in PLN. **Equities:** broad index moves affect funds according to their underlying holdings and tracking. **Commodities:** the futures symbol may not match spot prices or a fund's return because of roll costs.")
    st.caption("This screen currently charts market proxies. It does not automatically attribute every move to a confirmed news catalyst.")

elif page == "AI Analyst":
    st.title("AI portfolio analyst")
    st.write("A read-only research assistant grounded in your ledger and the latest available quote responses. AI analysis is not a substitute for verified data or professional advice.")
    with st.form("ai_analyst_form"):
        prompt = st.text_area(
            "What would you like to investigate?",
            value="Review my portfolio. Identify concentration, currency and diversification risks, explain the largest exposures, and list what data is missing. Separate observations from assumptions.",
            height=120,
            key="ai_analyst_prompt",
        )
        analyze_clicked = st.form_submit_button("Analyze portfolio", type="primary")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip() or "openrouter/free"
    if analyze_clicked:
        if not api_key:
            st.error("OPENROUTER_API_KEY is empty. Add your key in .env and restart Streamlit.")
        else:
            quoted = portfolio[[c for c in ["ticker","name","quantity","currency","price","price_date","status","market_value_base","weight_pct","cost_basis_base","unrealized_pl_base","unrealized_pl_pct"] if c in portfolio.columns]].replace({np.nan: None}).to_dict(orient="records") if not portfolio.empty else []
            ledger_summary = {k:v for k,v in analytics.items() if k != "positions"}
            payload = {"reporting_currency":base_ccy,"positions":quoted,"ledger_summary":ledger_summary,
                       "last_quote_fetch":st.session_state.last_refresh,"quote_provider":"Yahoo Finance via yfinance"}
            system = """You are a careful ETF portfolio research analyst. Provide educational analysis only, not individualized regulated financial advice. Treat supplied prices as latest available, potentially delayed and incomplete. Do not invent data, news, fund holdings, fees, or causality. Separate observations, interpretations, assumptions and uncertainties. Do not guarantee or confidently predict investment returns. Do not instruct the user to buy or sell a specific security as a certainty. Explain data gaps. The agent is read-only and must not claim it executed actions."""
            user = f"Question: {prompt}\n\nCurrent supplied portfolio data:\n{json.dumps(payload, default=str)}\n\nStructure your answer as: 1) executive summary; 2) measured portfolio facts; 3) plausible movement drivers and evidence gaps; 4) key risks; 5) questions or checks to consider. State if the quote data is insufficient."
            with st.spinner("Calling OpenRouter…"):
                try:
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json",
                                 "HTTP-Referer":"http://localhost:8501","X-Title":"Investment Copilot"},
                        json={"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"temperature":0.2},timeout=90)
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.markdown(answer)
                except requests.RequestException as exc:
                    st.error(f"OpenRouter request failed: {exc}")
                except (KeyError, IndexError, ValueError) as exc:
                    st.error(f"Unexpected OpenRouter response: {exc}")
    with st.expander("Data sent to the AI provider"):
        st.write("Only the selected question, current portfolio/ledger summary and displayed quote data are sent to OpenRouter when you run an analysis. Do not include sensitive personal details in the prompt.")

elif page == "Forecast Lab":
    st.title("Forecast Lab")
    st.write("Explore a range of possible outcomes under explicit assumptions. These are simulations, not predictions, and are sensitive to the inputs chosen.")
    if active_tickers and not metrics["valuation_complete"]:
        st.warning("At least one holding is missing price or FX data. The starting portfolio value defaults to zero; enter a complete, verified starting value manually before interpreting the simulation.")
    c1,c2,c3,c4 = st.columns(4)
    initial = c1.number_input(f"Current ETF value ({base_ccy})", min_value=0.0, value=float(metrics["value"] or 0), step=1000.0)
    monthly = c2.number_input(f"Monthly contribution ({base_ccy})", min_value=0.0, value=500.0, step=100.0)
    ann_return = c3.number_input("Assumed annual arithmetic return (%)", min_value=-50.0, max_value=50.0, value=5.0, step=0.5) / 100
    ann_vol = c4.number_input("Assumed annual volatility (%)", min_value=0.0, max_value=100.0, value=15.0, step=1.0) / 100
    inflation = st.slider("Annual inflation assumption (%)", min_value=0.0, max_value=15.0, value=2.5, step=0.25) / 100
    sims = st.select_slider("Number of simulations", options=[1000, 2000, 4000, 8000], value=4000)
    if st.button("Run simulation", type="primary"):
        st.session_state.forecast = simulate_forecast(initial, monthly, ann_return, ann_vol, inflation, simulations=int(sims), reporting_currency=base_ccy)
        st.session_state.forecast_assumptions = {"initial":initial,"monthly":monthly,"return":ann_return,"vol":ann_vol,"inflation":inflation,"sims":sims}
    forecast = st.session_state.get("forecast", pd.DataFrame())
    if not forecast.empty:
        european_dataframe(forecast, use_container_width=True, hide_index=True)
        chart_df = forecast.melt(id_vars=["Horizon (years)"], value_vars=["10th percentile (nominal)","Median (nominal)","90th percentile (nominal)"], var_name="Outcome", value_name="Value")
        fig = px.line(chart_df, x="Horizon (years)", y="Value", color="Outcome", markers=True, title=f"Simulated nominal outcomes ({base_ccy})")
        fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.info("The 10th and 90th percentiles are percentiles of this simulation, not confidence guarantees. The model does not know future regime changes, taxes, tracking error or actual market returns.")
    st.markdown("**Model assumptions:** independent monthly log-return draws with constant annualized return and volatility; contributions added at each month-end; no tax, TER, transaction cost or withdrawal modeling; inflation deflates nominal values. Update assumptions to explore sensitivity rather than treating one run as the expected result.")

elif page == "Data operations":
    st.title("Market-data operations")
    st.markdown('<div class="terminal-tag">PERSISTENT CACHE · SCHEDULED INGESTION · AUDIT TRAIL</div>', unsafe_allow_html=True)
    st.write("Current prices and currency-pair quotes, adjusted daily price histories, and historical daily FX reference rates are cached in the database. The separate worker refreshes held/watchlisted symbols, their FX pairs, and configured benchmark symbols on a schedule; it does not place trades.")
    status = market_data_cache_summary()
    a, b, c, d = st.columns(4)
    a.metric("Cached quote symbols", european_number(status["quote_rows"], decimals=0))
    b.metric("Historical observations", european_number(status["history_rows"], decimals=0))
    c.metric("FX fixing observations", european_number(status["fx_fixings"], decimals=0))
    d.metric("Funds with imported holdings", european_number(status["constituent_funds"], decimals=0))
    cache_updated_heading = f"Updated ({display_timezone})"
    cache_rows = [
        {"Data": "Adjusted price history", "Coverage": f"{status['history_first_date'] or '—'} to {status['history_last_date'] or '—'} · {status['history_rows']} observations", cache_updated_heading: "—"},
        {"Data": "Latest market quotes", "Coverage": f"{status['quote_rows']} cached quotes", cache_updated_heading: local_market_time(status["last_quote_fetch"])},
        {"Data": "FX reference fixings", "Coverage": f"{status['fx_fixings']} cached rates", cache_updated_heading: local_market_time(status["last_fx_cache_update"])},
    ]
    st.subheader("Market-data cache")
    european_dataframe(pd.DataFrame(cache_rows), use_container_width=True, hide_index=True, row_height=34, column_config={
        "Data":st.column_config.TextColumn(width="small"),
        "Coverage":st.column_config.TextColumn(width="medium"),
        cache_updated_heading:st.column_config.TextColumn(width="medium"),
    })
    st.subheader("Scheduled refresh history")
    runs = read_market_data_runs(50)
    if runs.empty:
        st.info("No background refresh has been recorded yet. Start the worker in a second VS Code terminal, or run it once to populate the data cache.")
    else:
        for timestamp_column in ["started_at", "finished_at"]:
            if timestamp_column in runs.columns:
                runs[timestamp_column] = runs[timestamp_column].map(local_market_time)
        runs = runs.rename(columns={"started_at": f"Started ({display_timezone})", "finished_at": f"Finished ({display_timezone})"})
        european_dataframe(runs, use_container_width=True, hide_index=True, row_height=34)
    st.subheader("Run the scheduled worker")
    st.markdown("""
**Local VS Code + WSL2**, after `./start.sh` has created `.venv`, open another terminal from the project root:

```bash
./scripts/run_market_data_worker.sh
```

For a one-off import/refresh:

```bash
./scripts/run_market_data_worker.sh --once
```

The worker runs once at startup, then repeats every 24 hours by default. Configure `ETF_MARKET_DATA_REFRESH_SECONDS`, `ETF_MARKET_DATA_HISTORY_PERIOD` and `ETF_MARKET_DATA_EXTRA_TICKERS` in `.env` to adjust the interval, history depth and benchmark/universe symbols. `docker compose up -d` also starts the persistent worker service. Historical observations are deduplicated by ticker/date and refreshes are written to the market-data run log.
""")
    st.caption("Market prices from Yahoo Finance and Frankfurter FX fixings are research/reference data, can be delayed or incomplete, and are not guaranteed executable prices. Review the price date and last successful refresh before relying on a figure.")

elif page == "Methodology & data":
    st.title("Methodology, data & security")
    st.markdown("""
### Accounting
- The SQLite transaction ledger is the source of truth. Quantities are derived from opening positions, buys and sells.
- Remaining acquisition cost uses a weighted-average-cost method in the selected reporting currency. A sell realizes proceeds net of transaction fee minus the proportional book cost.
- Purchase and sale fees affect book cost/proceeds. Standalone fee transactions are tracked separately.
- Dividends are recorded as income and added to the cash ledger. Reinvested dividends require an additional BUY transaction.
- OPENING lets you enter the current position and average acquisition price. If you do not know the price, enter zero; P/L will be marked as unavailable.
- Cash ledger balances are tracked by currency, but cash balances are not included in the headline ETF market value. Add cash-in/cash-out entries consistently if you use the cash ledger.

### Data
- **Prices and history:** Yahoo Finance via `yfinance`. Coverage, timestamp precision, adjustment conventions and availability vary by ticker. The fetch timestamp is not an exchange quote timestamp.
- **Current FX:** Yahoo Finance currency-pair quotes are used for current portfolio values and daily P/L where available. If a pair quote is unavailable, the latest Frankfurter/ECB reference rate is used.
- **Historical FX:** Frankfurter API, based on ECB daily reference rates for supported currencies/dates. These are not executable trading rates; rates can be unavailable on some dates/currencies.
- **Price currency:** The ETF's trading listing currency may differ from the currencies of the underlying assets and from the fund's base currency.
- **P/L:** Daily portfolio P/L includes changes in both the instrument price and current FX quote. Historical FX on transaction dates is used for book cost when available. Where FX is missing, cost basis and some metrics may be incomplete. The current engine does not perform tax-lot accounting, tax calculation, or broker-statement reconciliation.
- **Persistent cache:** Quote metadata, adjusted daily prices and historical FX observations are cached in SQLite. Price observations preserve the provider date and quote fetch timestamp; cache time does not prove the underlying price is live.
- **Snapshots:** Manual snapshots use the selected reporting currency. The scheduled worker records a PLN snapshot only when valuation is complete. Snapshots are point-in-time values, not returns.
- **Performance page:** The ETF-holdings return proxy chain-links weighted adjusted-close returns using quantities from the ledger. A selected benchmark is converted to the same reporting currency and compared over common dates. It excludes broker cash, taxes, withholding, non-ETF holdings and exact intraday trade-time effects. The separate XIRR is approximate and is not a broker-reconciled account return.
- **Data gaps:** Short forward-fills are limited to three observations for aligning exchange/FX holiday calendars; longer gaps stop a comparison segment. Historical risk calculations do not forward-fill the underlying histories.
- **Instrument master, watchlist and constituent sources:** User-managed metadata and symbols are stored in SQLite. Constituent files retain source name, optional source URL and as-of date. Overlap is computed from imported weights, not fabricated data; incomplete constituent snapshots understate overlap.
- **Scheduled worker:** `scripts/market_data_worker.py` periodically refreshes quotes and history. It is read-only and does not trade. Provider requests can fail or be throttled; inspect the run log under Data operations.
- **Forecasts:** Monte Carlo scenarios use fixed user inputs and simplified lognormal monthly returns. Real-value result labels follow the selected reporting currency. The assumptions do not account for changing volatility regimes, fat tails, valuation levels or guaranteed outcomes.

### Security
- Keep `.env` local and never commit secrets. `.gitignore` excludes it and the data directory.
- The AI analyst sends the selected prompt and current summary to OpenRouter only after you click Analyze. Do not place sensitive personal information in prompts.
- The application does not log in to a broker, place orders, store broker credentials, or execute trades.
- Before internet-facing deployment, add user authentication, per-user database isolation, secret management, HTTPS, backups and operational monitoring.
    """)
    st.code('OPENROUTER_API_KEY=""\nOPENROUTER_MODEL=openrouter/free', language="dotenv")
    st.subheader("Database backup")
    st.write("The database defaults to `data/portfolio.sqlite3`. The online backup script `./scripts/backup_db.sh` uses SQLite's backup API, including when WAL is enabled. Keep backups outside version control and test restoring a copy. Export transactions to CSV from the Transactions page for a readable ledger copy.")
