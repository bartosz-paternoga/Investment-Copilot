"""ETF look-through analytics that only use explicitly imported source data."""
from __future__ import annotations

import pandas as pd


def holdings_weight_coverage(holdings: pd.DataFrame) -> pd.DataFrame:
    """Report supplied constituent-weight totals by fund; no normalization occurs."""
    cols = ["fund_ticker", "as_of", "weight_total_pct", "constituents", "coverage_status", "source_name", "source_url"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (fund, as_of), group in holdings.groupby(["fund_ticker", "as_of"], dropna=False):
        total = float(pd.to_numeric(group["weight_pct"], errors="coerce").fillna(0).sum())
        if 90 <= total <= 110:
            status = "Near 100% — likely broad coverage"
        elif total < 90:
            status = "Partial/unknown coverage — overlap may be understated"
        else:
            status = "Over 110% — check duplicate or scale errors"
        rows.append({"fund_ticker": fund, "as_of": as_of, "weight_total_pct": total,
                     "constituents": int(group["constituent_ticker"].nunique()), "coverage_status": status,
                     "source_name": "; ".join(sorted(group["source_name"].dropna().astype(str).unique())),
                     "source_url": "; ".join(sorted(group["source_url"].dropna().astype(str).unique()))})
    return pd.DataFrame(rows, columns=cols).sort_values("fund_ticker", ignore_index=True)


def calculate_etf_overlap(holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise constituent overlap from supplied percentage weights.

    The overlap score is Σ min(weight_A, weight_B) over shared constituents. It is
    a standard weight-overlap proxy only when both source files represent complete
    fund holdings with weights on a 0–100 percentage-point scale. No renormalization
    is performed, so partial files are not made to look complete.
    """
    summary_cols = ["ETF A", "ETF B", "Overlap %", "Shared constituents"]
    detail_cols = ["ETF A", "ETF B", "Constituent", "Weight A %", "Weight B %", "Overlap contribution %"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)
    required = {"fund_ticker", "constituent_ticker", "weight_pct"}
    if not required.issubset(holdings.columns):
        return pd.DataFrame(columns=summary_cols), pd.DataFrame(columns=detail_cols)
    frame = holdings.copy()
    frame["fund_ticker"] = frame["fund_ticker"].astype(str).str.strip().str.upper()
    frame["constituent_ticker"] = frame["constituent_ticker"].astype(str).str.strip().str.upper()
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="coerce")
    frame = frame.dropna(subset=["weight_pct"])
    # A single overlap comparison must use one snapshot date per fund. The caller
    # normally supplies the latest snapshot; this additionally keeps the latest.
    if "as_of" in frame:
        frame = frame.sort_values("as_of").drop_duplicates(["fund_ticker", "constituent_ticker"], keep="last")
    funds = sorted(frame["fund_ticker"].unique())
    summaries, details = [], []
    for i, fund_a in enumerate(funds):
        a = frame.loc[frame["fund_ticker"] == fund_a].set_index("constituent_ticker")["weight_pct"]
        for fund_b in funds[i + 1:]:
            b = frame.loc[frame["fund_ticker"] == fund_b].set_index("constituent_ticker")["weight_pct"]
            shared = sorted(set(a.index).intersection(b.index))
            score = 0.0
            for constituent in shared:
                wa, wb = float(a.loc[constituent]), float(b.loc[constituent])
                contribution = min(wa, wb)
                score += contribution
                details.append({"ETF A": fund_a, "ETF B": fund_b, "Constituent": constituent,
                                "Weight A %": wa, "Weight B %": wb, "Overlap contribution %": contribution})
            summaries.append({"ETF A": fund_a, "ETF B": fund_b, "Overlap %": score,
                              "Shared constituents": len(shared)})
    summary = pd.DataFrame(summaries, columns=summary_cols)
    detail = pd.DataFrame(details, columns=detail_cols)
    if not summary.empty:
        summary = summary.sort_values("Overlap %", ascending=False, ignore_index=True)
    if not detail.empty:
        detail = detail.sort_values(["ETF A", "ETF B", "Overlap contribution %"], ascending=[True, True, False], ignore_index=True)
    return summary, detail
