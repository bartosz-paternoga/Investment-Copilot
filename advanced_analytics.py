"""Pure portfolio analytics helpers. No market-data calls are made here."""
from __future__ import annotations

from datetime import date
import math
import numpy as np
import pandas as pd


def xirr(cash_flows: list[tuple[str, float]], *, max_rate: float = 1000.0) -> float | None:
    """Annualized money-weighted return from dated investor cash flows.

    Contributions are negative, withdrawals/distributions positive, and the
    current portfolio value should be supplied as a positive terminal flow.
    Returns None when flows do not support a unique practical solution.
    """
    flows = []
    for d, amount in cash_flows:
        try:
            dt = date.fromisoformat(str(d)[:10])
            value = float(amount)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            flows.append((dt, value))
    if len(flows) < 2 or not any(v < 0 for _, v in flows) or not any(v > 0 for _, v in flows):
        return None
    start = min(d for d, _ in flows)
    days = np.array([(d - start).days for d, _ in flows], dtype=float)
    values = np.array([v for _, v in flows], dtype=float)

    def npv(rate: float) -> float:
        if rate <= -1:
            return float("inf")
        return float(np.sum(values / np.power(1.0 + rate, days / 365.0)))

    # Search a broad grid for a sign-changing bracket, then bisect it.
    grid = np.concatenate((np.linspace(-0.999, 1.0, 500), np.geomspace(2.0, max_rate + 1.0, 250) - 1.0))
    previous_rate, previous_value = float(grid[0]), npv(float(grid[0]))
    brackets = []
    for rate in grid[1:]:
        rate = float(rate)
        value = npv(rate)
        if math.isfinite(previous_value) and math.isfinite(value) and previous_value * value <= 0:
            brackets.append((previous_rate, rate))
        previous_rate, previous_value = rate, value
    if not brackets:
        return None
    # Prefer the root closest to zero; non-conventional cash flows can have multiple roots.
    lo, hi = min(brackets, key=lambda pair: abs((pair[0] + pair[1]) / 2))
    f_lo = npv(lo)
    for _ in range(160):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8 or abs(hi - lo) < 1e-10:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def portfolio_xirr(transactions: pd.DataFrame, terminal_value: float, as_of: str | None = None,
                   reporting_currency: str = "PLN", fx_converter=None) -> float | None:
    """Approximate money-weighted return from investment cash flows and terminal ETF value.

    OPENING/BUY are negative investment flows, SELL and DIVIDEND are positive
    proceeds/income, and current ETF market value is the terminal positive flow.
    Every event is converted to reporting currency on its transaction date when
    an FX converter is supplied. CASH_IN/CASH_OUT are deliberately excluded to
    avoid double-counting funding events alongside purchases/sales. This is not
    a full account return if cash balances, taxes, or other assets are material.
    """
    if transactions.empty or not np.isfinite(terminal_value) or terminal_value < 0:
        return None
    flows: list[tuple[str, float]] = []
    for _, tx in transactions.iterrows():
        kind = str(tx.get("transaction_type", "")).upper()
        currency = str(tx.get("currency", reporting_currency) or reporting_currency).upper()
        tx_date = str(tx.get("trade_date", ""))[:10]
        quantity = float(tx.get("quantity", 0) or 0)
        price = float(tx.get("unit_price", 0) or 0)
        amount = float(tx.get("amount", 0) or 0)
        fee = float(tx.get("fee", 0) or 0)
        if kind in {"OPENING", "BUY"}:
            value = -(quantity * price + fee)
        elif kind == "SELL":
            value = quantity * price - fee
        elif kind == "DIVIDEND":
            value = amount
        else:
            continue
        if fx_converter is not None:
            rate = fx_converter(currency, reporting_currency, tx_date)
            if rate is None or not np.isfinite(rate):
                return None
            value *= float(rate)
        flows.append((tx_date, value))
    flows.append((as_of or date.today().isoformat(), float(terminal_value)))
    return xirr(flows)


def historical_risk_table(history: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument price-return and risk summary with conservative missing-data handling."""
    rows = []
    if history.empty:
        return pd.DataFrame(columns=["Ticker", "Price return %", "Annualized volatility %", "Maximum drawdown %", "Worst daily move %", "Observations"])
    for ticker in history.columns:
        prices = pd.to_numeric(history[ticker], errors="coerce").dropna()
        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(prices) < 2 or len(returns) < 2:
            continue
        drawdown = prices / prices.cummax() - 1.0
        rows.append({"Ticker": ticker, "Price return %": (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
                     "Annualized volatility %": returns.std(ddof=1) * np.sqrt(252) * 100,
                     "Maximum drawdown %": drawdown.min() * 100,
                     "Worst daily move %": returns.min() * 100, "Observations": len(prices)})
    return pd.DataFrame(rows)
