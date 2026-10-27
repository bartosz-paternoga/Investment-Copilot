"""Historical ETF total-return calculations; no network or database calls live here.

Performance is based on adjusted close histories (dividends/splits reinvested by the
provider), historical FX where available, and transaction quantities. Cash balances,
taxes, withholding tax and brokerage costs outside recorded price/fee data are not
included in this ETF-holdings index.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np
import pandas as pd


def convert_prices_to_currency(
    local_prices: pd.DataFrame,
    currency_by_ticker: Mapping[str, str],
    fx_history_by_currency: Mapping[str, pd.Series],
    reporting_currency: str = "PLN",
    *,
    fx_forward_fill_rows: int = 3,
) -> pd.DataFrame:
    """Convert adjusted listing-currency prices into a reporting currency.

    FX series should be source-currency -> reporting-currency reference rates.
    Gaps are forward-filled only for a small, explicit number of observations to
    accommodate non-overlapping exchange/FX holidays; longer gaps remain missing.
    """
    if local_prices.empty:
        return pd.DataFrame(index=local_prices.index)
    prices = local_prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    target = str(reporting_currency or "PLN").upper()
    result = pd.DataFrame(index=prices.index)
    for raw_ticker in prices.columns:
        ticker = str(raw_ticker).upper()
        currency = str(currency_by_ticker.get(ticker, "") or "").upper()
        values = pd.to_numeric(prices[raw_ticker], errors="coerce")
        # Align listing calendars over short holidays only; get_history itself keeps
        # missing values intact so risk calculations never inherit unlimited fills.
        if int(fx_forward_fill_rows) > 0:
            values = values.ffill(limit=int(fx_forward_fill_rows))
        if currency == target:
            result[ticker] = values
            continue
        fx = fx_history_by_currency.get(currency)
        if not currency or fx is None or len(fx) == 0:
            result[ticker] = np.nan
            continue
        fx = pd.to_numeric(fx, errors="coerce").copy()
        fx.index = pd.to_datetime(fx.index).tz_localize(None).normalize()
        fx = fx[~fx.index.duplicated(keep="last")].sort_index()
        aligned = fx.reindex(prices.index).ffill(limit=max(0, int(fx_forward_fill_rows)))
        result[ticker] = values * aligned
    return result.replace([np.inf, -np.inf], np.nan)


def _quantity_events(transactions: pd.DataFrame) -> list[tuple[pd.Timestamp, str, float]]:
    events: list[tuple[pd.Timestamp, str, float]] = []
    if transactions is None or transactions.empty:
        return events
    required = {"trade_date", "ticker", "transaction_type", "quantity"}
    if not required.issubset(transactions.columns):
        return events
    txs = transactions.copy()
    txs["_date"] = pd.to_datetime(txs["trade_date"], errors="coerce").dt.normalize()
    txs = txs.dropna(subset=["_date"]).sort_values(["_date"] + (["id"] if "id" in txs else []))
    for _, tx in txs.iterrows():
        kind = str(tx.get("transaction_type", "")).upper()
        ticker = str(tx.get("ticker", "") or "").strip().upper()
        try:
            qty = float(tx.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        if ticker and np.isfinite(qty) and qty > 0 and kind in {"OPENING", "BUY", "SELL"}:
            events.append((tx["_date"], ticker, -qty if kind == "SELL" else qty))
    return events


def portfolio_total_return(
    base_currency_prices: pd.DataFrame,
    transactions: pd.DataFrame,
    *,
    max_stale_rows: int = 0,
) -> pd.DataFrame:
    """Build a chain-linked ETF-holdings total-return index and daily attribution.

    Daily returns use quantities held at the preceding close, so a buy/sell does
    not itself appear as investment profit. Transactions on a date take effect
    after that date's return calculation. An adjusted-close series is interpreted
    as a reinvested-dividend total-return proxy. Cash is excluded and should be
    disclosed in the UI.
    """
    if base_currency_prices is None or base_currency_prices.empty:
        return pd.DataFrame(columns=["daily_return", "portfolio_index", "coverage_count"])
    prices = base_currency_prices.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    prices.columns = [str(c).upper() for c in prices.columns]
    prices = prices.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(max_stale_rows) > 0:
        prices = prices.ffill(limit=int(max_stale_rows))
    events = _quantity_events(transactions)
    # Keep every ledger ticker, including symbols whose history could not be
    # downloaded. Silently dropping those positions would overstate coverage;
    # they instead make the portfolio return unavailable for affected intervals.
    quantities: dict[str, float] = defaultdict(float)
    first_date = prices.index[0]
    event_index = 0
    while event_index < len(events) and events[event_index][0] < first_date:
        _, ticker, delta = events[event_index]
        quantities[ticker] += delta
        event_index += 1

    rows: list[dict[str, float]] = []
    dates = list(prices.index)
    previous_values: dict[str, float] | None = None
    for idx, current_date in enumerate(dates):
        row: dict[str, float] = {"daily_return": np.nan, "coverage_count": 0.0}
        contributions: dict[str, float] = {}
        active = {ticker: qty for ticker, qty in quantities.items() if qty > 1e-10}
        if idx > 0 and active:
            prev = prices.iloc[idx - 1]
            cur = prices.iloc[idx]
            starting_values: dict[str, float] = {}
            asset_returns: dict[str, float] = {}
            for ticker, qty in active.items():
                p0, p1 = prev.get(ticker, np.nan), cur.get(ticker, np.nan)
                if pd.notna(p0) and pd.notna(p1) and float(p0) > 0 and float(p1) > 0:
                    starting_values[ticker] = float(p0) * qty
                    asset_returns[ticker] = float(p1) / float(p0) - 1.0
            # Missing a held ETF's prices is not silently treated as a zero return.
            if len(starting_values) == len(active) and sum(starting_values.values()) > 0:
                total_start = sum(starting_values.values())
                contributions = {
                    ticker: (starting_values[ticker] / total_start) * asset_returns[ticker]
                    for ticker in active
                }
                row["daily_return"] = float(sum(contributions.values()))
                row["coverage_count"] = float(len(active))
        for ticker, value in contributions.items():
            row[f"contribution::{ticker}"] = value
        # Events at or before this close affect the next return interval, not today's.
        while event_index < len(events) and events[event_index][0] <= current_date:
            _, ticker, delta = events[event_index]
            quantities[ticker] += delta
            if abs(quantities[ticker]) < 1e-10:
                quantities[ticker] = 0.0
            event_index += 1
        rows.append(row)
    result = pd.DataFrame(rows, index=prices.index)
    result.index.name = "date"
    valid_returns = result["daily_return"].dropna()
    if valid_returns.empty:
        result["portfolio_index"] = np.nan
        return result
    # Do not bridge an unknown return with an implicit zero. The index resumes at
    # 100 on the first valid observation after a gap and labels the gap explicitly.
    index_values = pd.Series(np.nan, index=result.index, dtype=float)
    current_index: float | None = None
    for dt, ret in result["daily_return"].items():
        if pd.isna(ret):
            current_index = None
            continue
        current_index = 100.0 * (1 + float(ret)) if current_index is None else current_index * (1 + float(ret))
        index_values.loc[dt] = current_index
    result["portfolio_index"] = index_values
    return result


def compare_to_benchmark(portfolio_daily: pd.DataFrame, benchmark_base_prices: pd.Series) -> pd.DataFrame:
    """Compare portfolio and benchmark returns in a common currency and date set."""
    if portfolio_daily is None or portfolio_daily.empty or benchmark_base_prices is None or benchmark_base_prices.empty:
        return pd.DataFrame(columns=["Portfolio index", "Benchmark index"])
    p_all = pd.to_numeric(portfolio_daily["daily_return"], errors="coerce").copy()
    # Stop at the first missing daily return after the series starts. Bridging a
    # long price/FX gap would imply a continuous performance record we do not have.
    valid_positions = np.flatnonzero(p_all.notna().to_numpy())
    if len(valid_positions) == 0:
        return pd.DataFrame(columns=["Portfolio return", "Benchmark return", "Portfolio index", "Benchmark index"])
    start_pos = int(valid_positions[0])
    p_segment = p_all.iloc[start_pos:]
    gaps = np.flatnonzero(p_segment.isna().to_numpy())
    if len(gaps):
        p_segment = p_segment.iloc[:int(gaps[0])]
    p = p_segment.dropna()
    b = pd.to_numeric(benchmark_base_prices, errors="coerce").copy()
    b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
    b = b[~b.index.duplicated(keep="last")].sort_index().pct_change(fill_method=None)
    common = p.index.intersection(b.index)
    paired = pd.DataFrame({"Portfolio return": p.reindex(common), "Benchmark return": b.reindex(common)})
    paired = paired.dropna(subset=["Portfolio return"])
    if paired.empty:
        return pd.DataFrame(columns=["Portfolio return", "Benchmark return", "Portfolio index", "Benchmark index"])
    # The benchmark can start a little later than a recently listed ETF. Ignore
    # leading unmatched dates, then stop at the first internal data gap.
    valid_pair = paired["Benchmark return"].notna().to_numpy()
    valid_pair_positions = np.flatnonzero(valid_pair)
    if len(valid_pair_positions) == 0:
        return pd.DataFrame(columns=["Portfolio return", "Benchmark return", "Portfolio index", "Benchmark index"])
    paired = paired.iloc[int(valid_pair_positions[0]):]
    benchmark_missing = np.flatnonzero(paired["Benchmark return"].isna().to_numpy())
    if len(benchmark_missing):
        paired = paired.iloc[:int(benchmark_missing[0])]
    paired = paired.dropna(subset=["Benchmark return"])
    if paired.empty:
        return pd.DataFrame(columns=["Portfolio return", "Benchmark return", "Portfolio index", "Benchmark index"])
    paired["Portfolio index"] = 100 * (1 + paired["Portfolio return"]).cumprod()
    paired["Benchmark index"] = 100 * (1 + paired["Benchmark return"]).cumprod()
    paired.index.name = "date"
    return paired


def linked_attribution(performance: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Geometrically link daily ticker contributions using the Carino method.

    Returns a table whose contributions approximately sum to the portfolio return
    over the selected valid interval. This is contribution attribution, not causal
    factor attribution or a Brinson benchmark attribution model.
    """
    if performance is None or performance.empty or "daily_return" not in performance:
        return pd.DataFrame(columns=["Ticker", "Contribution %"])
    frame = performance.copy()
    if start_date:
        frame = frame.loc[pd.to_datetime(frame.index) >= pd.Timestamp(start_date)]
    contribution_cols = [c for c in frame.columns if c.startswith("contribution::")]
    valid_mask = frame["daily_return"].notna().to_numpy()
    valid_positions = np.flatnonzero(valid_mask)
    if len(valid_positions):
        start_pos = int(valid_positions[0])
        segment = frame.iloc[start_pos:]
        gaps = np.flatnonzero(segment["daily_return"].isna().to_numpy())
        frame = segment.iloc[:int(gaps[0])] if len(gaps) else segment
    else:
        frame = frame.iloc[0:0]
    if frame.empty or not contribution_cols:
        return pd.DataFrame(columns=["Ticker", "Contribution %"])
    daily_returns = frame["daily_return"].astype(float).to_numpy()
    total_return = float(np.prod(1 + daily_returns) - 1)
    def k_factor(rate: float) -> float:
        if abs(rate) < 1e-10:
            return 1.0
        if rate <= -1:
            return np.nan
        return float(np.log1p(rate) / rate)
    total_k = k_factor(total_return)
    if not np.isfinite(total_k) or abs(total_k) < 1e-12:
        return pd.DataFrame(columns=["Ticker", "Contribution %"])
    daily_k = np.array([k_factor(float(r)) for r in daily_returns])
    records = []
    for col in contribution_cols:
        daily_c = pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy()
        linked = float(np.sum(daily_c * daily_k) / total_k)
        records.append({"Ticker": col.split("::", 1)[1], "Contribution %": linked * 100})
    out = pd.DataFrame(records).sort_values("Contribution %", ascending=False, ignore_index=True)
    return out


def performance_summary(performance: pd.DataFrame) -> dict[str, float | int | None]:
    """Return simple cumulative and annualized risk statistics for a return series."""
    if performance is None or performance.empty or "daily_return" not in performance:
        return {"total_return": None, "annualized_return": None, "annualized_volatility": None,
                "max_drawdown": None, "observations": 0}
    raw_returns = pd.to_numeric(performance["daily_return"], errors="coerce")
    valid_positions = np.flatnonzero(raw_returns.notna().to_numpy())
    if len(valid_positions):
        segment = raw_returns.iloc[int(valid_positions[0]):]
        gaps = np.flatnonzero(segment.isna().to_numpy())
        returns = segment.iloc[:int(gaps[0])] if len(gaps) else segment
    else:
        returns = raw_returns.iloc[0:0]
    if returns.empty:
        return {"total_return": None, "annualized_return": None, "annualized_volatility": None,
                "max_drawdown": None, "observations": 0}
    wealth = (1 + returns).cumprod()
    total = float(wealth.iloc[-1] - 1)
    valid_dates = pd.to_datetime(returns.index)
    years = max((valid_dates[-1] - valid_dates[0]).days / 365.25, 1 / 252)
    annualized = float(wealth.iloc[-1] ** (1 / years) - 1) if wealth.iloc[-1] > 0 else np.nan
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    drawdown = wealth / wealth.cummax() - 1
    return {"total_return": total, "annualized_return": annualized,
            "annualized_volatility": volatility if np.isfinite(volatility) else None,
            "max_drawdown": float(drawdown.min()), "observations": int(len(returns))}
