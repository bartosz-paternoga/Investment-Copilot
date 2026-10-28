"""ETF look-through analytics from dated, sourced constituent snapshots."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


_COUNTRY_NAMES = {
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "BE": "Belgium", "BR": "Brazil",
    "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CN": "China", "CO": "Colombia",
    "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "United Kingdom", "GR": "Greece", "HK": "Hong Kong", "HU": "Hungary",
    "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India", "IT": "Italy",
    "JP": "Japan", "KR": "South Korea", "LU": "Luxembourg", "MX": "Mexico", "MY": "Malaysia",
    "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines", "PL": "Poland",
    "PT": "Portugal", "SA": "Saudi Arabia", "SE": "Sweden", "SG": "Singapore", "TH": "Thailand",
    "TW": "Taiwan", "US": "United States", "VN": "Vietnam", "ZA": "South Africa",
}
_COUNTRY_REGIONS = {
    **{code: "North America" for code in ("CA", "MX", "US")},
    **{code: "Latin America" for code in ("AR", "BR", "CL", "CO")},
    **{code: "Europe" for code in ("AT", "BE", "CH", "CZ", "DE", "DK", "ES", "FI", "FR", "GB", "GR", "HU", "IE", "IT", "LU", "NL", "NO", "PL", "PT", "SE")},
    **{code: "Asia-Pacific" for code in ("AU", "CN", "HK", "ID", "IN", "JP", "KR", "MY", "NZ", "PH", "SG", "TH", "TW", "VN")},
    **{code: "Middle East & Africa" for code in ("IL", "SA", "ZA")},
}


def _company_identity(name: str, ticker: str) -> str:
    """Match listings for the same company across issuer naming conventions."""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[A-Z0-9]+", text.upper())
    # Issuers often vary only in the legal suffix; removing it lets, for example,
    # "ASML Holding NV" and "ASML HOLDING" resolve to the same company.
    legal_suffixes = {"AG", "BV", "CORP", "CORPORATION", "INC", "LIMITED", "LTD", "NV", "PLC", "SA", "SE", "SPA"}
    while tokens and tokens[-1] in legal_suffixes:
        tokens.pop()
    return "name:" + " ".join(tokens) if tokens else "ticker:" + str(ticker or "").strip().upper()


def _region_for_country(country: str) -> str:
    value = str(country or "").strip()
    code = value.upper()
    if len(code) != 2:
        code = next((iso for iso, name in _COUNTRY_NAMES.items() if name.casefold() == value.casefold()), "")
    return _COUNTRY_REGIONS.get(code, "Other / unknown")


def calculate_portfolio_xray(portfolio: pd.DataFrame, holdings: pd.DataFrame) -> dict:
    """Build sourced ETF look-through exposures without inflating partial files.

    Constituent weights are percentage points. Partial snapshots contribute only
    their reported weights; a small total above 100% is scaled back to 100%, and
    snapshots above 110% are excluded as likely data errors.
    """
    company_columns = ["constituent_ticker", "constituent_name", "exposure_value", "portfolio_weight_pct", "etf_count", "held_through"]
    category_columns = ["category", "exposure_value", "portfolio_weight_pct", "constituents", "etfs"]
    coverage_columns = ["Fund", "Position value", "Snapshot weight %", "Look-through value", "As of", "Source", "Coverage"]
    empty = {
        "companies": pd.DataFrame(columns=company_columns),
        "sectors": pd.DataFrame(columns=category_columns),
        "countries": pd.DataFrame(columns=category_columns),
        "regions": pd.DataFrame(columns=category_columns),
        "coverage": pd.DataFrame(columns=coverage_columns),
        "portfolio_value": 0.0,
        "covered_value": 0.0,
        "coverage_pct": 0.0,
        "priced_positions": 0,
        "covered_positions": 0,
    }
    if portfolio is None or portfolio.empty or not {"ticker", "market_value_base"}.issubset(portfolio.columns):
        return empty

    positions = portfolio[["ticker", "market_value_base"]].copy()
    positions["ticker"] = positions["ticker"].fillna("").astype(str).str.strip().str.upper()
    positions["position_value"] = pd.to_numeric(positions.pop("market_value_base"), errors="coerce")
    positions = positions[(positions["ticker"] != "") & positions["position_value"].gt(0)].dropna(subset=["position_value"])
    if positions.empty:
        return empty
    positions = positions.groupby("ticker", as_index=False)["position_value"].sum()
    portfolio_value = float(positions["position_value"].sum())

    if holdings is None or holdings.empty or not {"fund_ticker", "constituent_ticker", "weight_pct"}.issubset(holdings.columns):
        result = dict(empty)
        result["portfolio_value"] = portfolio_value
        result["priced_positions"] = len(positions)
        result["coverage"] = pd.DataFrame([
            {"Fund": row.ticker, "Position value": row.position_value, "Snapshot weight %": 0.0,
             "Look-through value": 0.0, "As of": "—", "Source": "—", "Coverage": "No imported snapshot"}
            for row in positions.itertuples(index=False)
        ], columns=coverage_columns)
        return result

    source = holdings.copy()
    source["fund_ticker"] = source["fund_ticker"].fillna("").astype(str).str.strip().str.upper()
    source["constituent_ticker"] = source["constituent_ticker"].fillna("").astype(str).str.strip().str.upper()
    source["weight_pct"] = pd.to_numeric(source["weight_pct"], errors="coerce")
    source = source.dropna(subset=["weight_pct"])
    if "as_of" not in source:
        source["as_of"] = ""
    source["as_of"] = source["as_of"].fillna("").astype(str)
    source = source.sort_values("as_of").drop_duplicates(["fund_ticker", "constituent_ticker"], keep="last")
    for column in ("constituent_name", "sector", "country", "region", "source_name", "source_url"):
        if column not in source:
            source[column] = ""
        source[column] = source[column].fillna("").astype(str).str.strip()

    position_values = positions.set_index("ticker")["position_value"].to_dict()
    coverage_rows, exposure_rows = [], []
    covered_value = 0.0
    covered_positions = 0
    for fund, position_value in position_values.items():
        fund_rows = source[source["fund_ticker"] == fund].copy()
        if fund_rows.empty:
            coverage_rows.append({"Fund": fund, "Position value": position_value, "Snapshot weight %": 0.0,
                                  "Look-through value": 0.0, "As of": "—", "Source": "—",
                                  "Coverage": "No imported snapshot"})
            continue
        source_weight = float(fund_rows["weight_pct"].clip(lower=0).sum())
        as_of_values = sorted(value for value in fund_rows["as_of"].unique() if value)
        sources = sorted(value for value in fund_rows["source_name"].unique() if value)
        if source_weight > 110:
            coverage_rows.append({"Fund": fund, "Position value": position_value,
                                  "Snapshot weight %": source_weight, "Look-through value": 0.0,
                                  "As of": ", ".join(as_of_values) or "—",
                                  "Source": ", ".join(sources) or "—",
                                  "Coverage": "Excluded: snapshot exceeds 110%"})
            continue
        scale = 100.0 / source_weight if source_weight > 100 else 1.0
        effective_weight = min(source_weight, 100.0)
        fund_covered_value = position_value * effective_weight / 100.0
        covered_value += fund_covered_value
        covered_positions += 1
        if source_weight < 90:
            coverage_label = "Partial; reported weights used as-is"
        elif source_weight > 100:
            coverage_label = "Near full; small total excess scaled to 100%"
        else:
            coverage_label = "Near full coverage"
        coverage_rows.append({"Fund": fund, "Position value": position_value,
                              "Snapshot weight %": source_weight, "Look-through value": fund_covered_value,
                              "As of": ", ".join(as_of_values) or "—",
                              "Source": ", ".join(sources) or "—", "Coverage": coverage_label})
        fund_rows["effective_weight_pct"] = fund_rows["weight_pct"].clip(lower=0) * scale
        fund_rows["exposure_value"] = position_value * fund_rows["effective_weight_pct"] / 100.0
        for row in fund_rows.itertuples(index=False):
            if not row.constituent_ticker or row.exposure_value <= 0:
                continue
            exposure_rows.append({
                "constituent_ticker": row.constituent_ticker,
                "constituent_name": row.constituent_name or row.constituent_ticker,
                "sector": row.sector,
                "country": row.country,
                "region": row.region,
                "fund_ticker": fund,
                "exposure_value": float(row.exposure_value),
            })

    companies = pd.DataFrame(exposure_rows)
    if companies.empty:
        company_summary = pd.DataFrame(columns=company_columns)
        sector_summary = pd.DataFrame(columns=category_columns)
        country_summary = pd.DataFrame(columns=category_columns)
        region_summary = pd.DataFrame(columns=category_columns)
    else:
        companies["_company_key"] = companies.apply(
            lambda row: _company_identity(row["constituent_name"], row["constituent_ticker"]), axis=1
        )
        company_summary = companies.groupby("_company_key", as_index=False).agg(
            constituent_ticker=("constituent_ticker", lambda values: ", ".join(sorted(set(v for v in values if v)))),
            constituent_name=("constituent_name", lambda values: next((v for v in values if v), "")),
            exposure_value=("exposure_value", "sum"),
            etf_count=("fund_ticker", "nunique"),
            held_through=("fund_ticker", lambda values: ", ".join(sorted(set(values)))),
        )
        company_summary["portfolio_weight_pct"] = company_summary["exposure_value"] / portfolio_value * 100.0
        company_summary = company_summary[company_columns].sort_values("exposure_value", ascending=False, ignore_index=True)

        def summarize_category(column: str) -> pd.DataFrame:
            categorized = companies[companies[column].str.strip() != ""].copy()
            if categorized.empty:
                return pd.DataFrame(columns=category_columns)
            if column == "country":
                categorized[column] = categorized[column].map(
                    lambda value: _COUNTRY_NAMES.get(str(value).strip().upper(), str(value).strip())
                )
            elif column == "region":
                categorized[column] = categorized.apply(
                    lambda row: row[column].strip() or _region_for_country(row["country"]),
                    axis=1,
                )
            categorized["_category_key"] = categorized[column].str.casefold()
            summary = categorized.groupby("_category_key", as_index=False).agg(
                category=(column, lambda values: next((v for v in values if v), "")),
                exposure_value=("exposure_value", "sum"),
                constituents=("_company_key", "nunique"),
                etfs=("fund_ticker", "nunique"),
            )
            summary["portfolio_weight_pct"] = summary["exposure_value"] / portfolio_value * 100.0
            return summary[category_columns].sort_values("exposure_value", ascending=False, ignore_index=True)

        sector_summary = summarize_category("sector")
        country_summary = summarize_category("country")
        region_summary = summarize_category("region")

    coverage = pd.DataFrame(coverage_rows, columns=coverage_columns)
    coverage_pct = covered_value / portfolio_value * 100.0 if portfolio_value > 0 else 0.0
    return {
        "companies": company_summary,
        "sectors": sector_summary,
        "countries": country_summary,
        "regions": region_summary,
        "coverage": coverage,
        "portfolio_value": portfolio_value,
        "covered_value": covered_value,
        "coverage_pct": coverage_pct,
        "priced_positions": len(positions),
        "covered_positions": covered_positions,
    }


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
