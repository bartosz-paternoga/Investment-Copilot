"""Build dated, source-linked evidence for the optional AI portfolio analyst."""
from __future__ import annotations

import re
from urllib.parse import quote
from typing import Any

import numpy as np
import pandas as pd

from research_analytics import calculate_portfolio_xray


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_analyst_evidence(
    portfolio: pd.DataFrame,
    quotes: pd.DataFrame,
    issuer_holdings: pd.DataFrame,
    fx_metadata: dict[str, dict[str, Any]],
    reporting_currency: str,
    as_of: str,
    focus_query: str = "",
) -> dict[str, Any]:
    """Create a small evidence packet and deterministic scenario for the model."""
    currency = str(reporting_currency or "PLN").upper()
    packet: dict[str, Any] = {
        "briefing_as_of": as_of,
        "reporting_currency": currency,
        "facts": [],
        "potential_risks": [],
        "positions": [],
        "underlying_company_exposures": [],
        "focus_company_exposures": [],
        "focus_company_query": str(focus_query or "").strip(),
        "focus_company_lookup": {"attempted": False, "terms": [], "match_count": 0},
        "xray_coverage_pct": 0.0,
        "listing_currency_exposures": [],
        "country_exposures": [],
        "sector_exposures": [],
        "region_exposures": [],
        "other_asset_exposures": [],
        "issuer_holdings_coverage": [],
        "stress_scenario": None,
        "data_gaps": [],
        "sources": [],
    }

    source_rows: list[dict[str, Any]] = []
    quote_ref_by_ticker: dict[str, str] = {}
    fx_ref_by_currency: dict[str, str] = {}
    holding_refs_by_fund: dict[str, list[str]] = {}

    def add_source(ref: str, name: str, source_date: Any = None, url: Any = None, detail: str = "") -> None:
        source_date = _clean(source_date)
        url = _clean(url)
        source_rows.append({
            "Reference": ref,
            "Source": str(_clean(name) or "Source not recorded"),
            "As of": str(source_date or "Not supplied"),
            "Details": str(_clean(detail) or ""),
            "URL": str(url or ""),
        })

    add_source(
        "L01",
        "Private portfolio ledger (app)",
        as_of,
        "",
        "App-calculated current positions and transaction-derived cost basis; only the resulting position summary is sent for this analysis.",
    )

    quote_frame = quotes.copy() if isinstance(quotes, pd.DataFrame) else pd.DataFrame()
    if not quote_frame.empty and "ticker" in quote_frame:
        quote_frame["ticker"] = quote_frame["ticker"].fillna("").astype(str).str.upper()
        quote_frame = quote_frame.drop_duplicates("ticker", keep="last").set_index("ticker")

    positions = portfolio.copy() if isinstance(portfolio, pd.DataFrame) else pd.DataFrame()
    if positions.empty:
        packet["data_gaps"].append({"item": "Portfolio positions", "gap": "No open positions are available for this briefing"})
    if not positions.empty:
        positions["ticker"] = positions["ticker"].fillna("").astype(str).str.strip().str.upper()
        positions["market_value_base"] = pd.to_numeric(positions["market_value_base"], errors="coerce")
        priced_total_value = positions["market_value_base"].sum(min_count=1)
        priced_total = float(priced_total_value) if pd.notna(priced_total_value) else 0.0
    else:
        priced_total = 0.0
    if not np.isfinite(priced_total):
        priced_total = 0.0

    positions_payload = []
    missing_cost_basis: list[str] = []
    priced_positions = positions[positions["market_value_base"].notna()].copy() if not positions.empty else pd.DataFrame()
    if not priced_positions.empty and priced_total > 0:
        priced_positions["briefing_weight_pct"] = priced_positions["market_value_base"] / priced_total * 100.0

    for index, row in enumerate(positions.itertuples(index=False), start=1):
        ticker = str(getattr(row, "ticker", "") or "").upper()
        quote_ref = quote_ref_by_ticker.get(ticker)
        price_quote = quote_frame.loc[ticker] if ticker and ticker in quote_frame.index else None
        if quote_ref is None:
            quote_ref = f"P{len(quote_ref_by_ticker) + 1:02d}"
            quote_ref_by_ticker[ticker] = quote_ref
            quote_date = getattr(row, "price_date", None)
            fetched = _clean(price_quote.get("fetched_at")) if price_quote is not None else None
            quote_timing = _clean(getattr(row, "price_timing", None)) or ""
            quote_source = "Yahoo Finance via yfinance"
            quote_status = "No quote response in current data"
            quote_error = ""
            if price_quote is not None:
                quote_date = _clean(price_quote.get("price_date")) or quote_date
                quote_timing = _clean(price_quote.get("price_timing")) or quote_timing
                quote_source = str(_clean(price_quote.get("source")) or quote_source)
                quote_status = str(_clean(price_quote.get("status")) or "Status not supplied")
                quote_error = str(_clean(price_quote.get("error")) or "")
            add_source(
                quote_ref,
                f"{quote_source} · {ticker}",
                quote_date,
                f"https://finance.yahoo.com/quote/{quote(ticker, safe='.^=-')}/" if ticker else "",
                f"Status: {quote_status}; quote fetch timestamp (UTC): {fetched or 'not available'}; displayed timing: {quote_timing or 'not available'}; provider note: {quote_error or 'none'}",
            )

        instrument_currency = str(_clean(getattr(row, "currency", None)) or "").upper()
        fx_ref = None
        if instrument_currency and instrument_currency != currency:
            fx_ref = fx_ref_by_currency.get(instrument_currency)
            fx_info = fx_metadata.get(instrument_currency, {})
            if fx_ref is None:
                fx_ref = f"X{len(fx_ref_by_currency) + 1:02d}"
                fx_ref_by_currency[instrument_currency] = fx_ref
                kind = str(_clean(fx_info.get("kind")) or "reference")
                fx_ticker = str(_clean(fx_info.get("ticker")) or "")
                if kind == "spot" and fx_ticker:
                    fx_name = "Yahoo Finance spot FX via yfinance"
                    fx_url = f"https://finance.yahoo.com/quote/{quote(fx_ticker, safe='.^=-')}/"
                else:
                    fx_name = "Frankfurter daily reference FX rate"
                    returned_date = str(fx_info.get("returned_date", "") or "latest")
                    fx_url = (
                        f"https://api.frankfurter.dev/v1/{quote(returned_date)}"
                        f"?base={quote(instrument_currency)}&symbols={quote(currency)}"
                    )
                add_source(
                    fx_ref,
                    f"{fx_name} · {instrument_currency}/{currency}",
                    fx_info.get("returned_date"),
                    fx_url,
                    f"FX type: {kind}; rate: {_clean(fx_info.get('rate'))}; fetch timestamp (UTC): {_clean(fx_info.get('fetched_at')) or 'not available'}",
                )

        market_value = _clean(getattr(row, "market_value_base", None))
        position_weight = (float(market_value) / priced_total * 100.0) if market_value is not None and priced_total > 0 else None
        price = _clean(getattr(row, "price", None))
        fx_rate = _clean(getattr(row, "fx_to_base", None))
        position = {
            "ticker": ticker,
            "name": _clean(getattr(row, "name", "")),
            "listing_currency": instrument_currency,
            "units": _clean(getattr(row, "quantity", None)),
            "latest_price": price,
            "price_date": _clean(getattr(row, "price_date", None)),
            "price_timing_local": _clean(getattr(row, "price_timing", None)),
            "fx_to_reporting_currency": fx_rate,
            "fx_timing_local": _clean(getattr(row, "fx_timing", None)),
            "market_value_reporting_currency": market_value,
            "share_of_priced_positions_pct": position_weight,
            "cost_basis_reporting_currency": _clean(getattr(row, "cost_basis_base", None)),
            "unrealized_pl_reporting_currency": _clean(getattr(row, "unrealized_pl_base", None)),
            "ledger_source_ref": "L01",
            "price_source_ref": quote_ref,
            "fx_source_ref": fx_ref or "Same as reporting currency",
        }
        positions_payload.append(position)

        if market_value is None:
            missing = "price or FX valuation unavailable"
            if price is None:
                missing = "price unavailable"
            elif fx_rate is None:
                missing = "FX rate unavailable"
            packet["data_gaps"].append({"item": ticker, "gap": missing})
        if _clean(getattr(row, "cost_basis_base", None)) is None:
            missing_cost_basis.append(ticker)

    if missing_cost_basis:
        packet["data_gaps"].append({
            "item": "Cost basis",
            "gap": f"Not available for {', '.join(missing_cost_basis)}; profit/loss cannot be measured for these positions",
        })

    packet["positions"] = positions_payload
    complete = bool(portfolio.attrs.get("valuation_complete", False)) if isinstance(portfolio, pd.DataFrame) else False
    packet["facts"].append({
        "ID": "F01",
        "Fact": "Value of priced positions",
        "Value": priced_total,
        "Unit": currency,
        "Basis": f"Sum of positions with a current price and FX value; valuation complete: {complete}",
        "Source refs": ", ".join(dict.fromkeys([
            "L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values(),
        ])) or "None",
    })
    packet["facts"].append({
        "ID": "F02",
        "Fact": "Current valuation coverage",
        "Value": f"{int(positions['market_value_base'].notna().sum()) if not positions.empty else 0} of {len(positions)} positions",
        "Unit": "positions",
        "Basis": "A position is valued only if current price and FX are available",
        "Source refs": ", ".join(dict.fromkeys(["L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values()])) or "None",
    })

    if not priced_positions.empty and priced_total > 0:
        currency_values = priced_positions.groupby("currency", dropna=False)["market_value_base"].sum().sort_values(ascending=False)
        currency_exposures = []
        for listing_currency, value in currency_values.items():
            currency_code = str(_clean(listing_currency) or "Unknown").upper()
            currency_tickers = priced_positions.loc[
                priced_positions["currency"].fillna("Unknown").astype(str).str.upper() == currency_code, "ticker"
            ].astype(str)
            currency_refs = ["L01"]
            currency_refs.extend(quote_ref_by_ticker.get(ticker, "") for ticker in currency_tickers)
            if currency_code != currency:
                currency_refs.append(fx_ref_by_currency.get(currency_code, ""))
            currency_exposures.append({
                "listing_currency": currency_code,
                "exposure_value_reporting_currency": float(value),
                "share_of_priced_positions_pct": float(value) / priced_total * 100.0,
                "source_refs": list(dict.fromkeys(ref for ref in currency_refs if ref)),
            })
        packet["listing_currency_exposures"] = currency_exposures[:10]
        largest_currency = currency_exposures[0]
        packet["facts"].append({
            "ID": "F08",
            "Fact": f"Largest listing-currency position share: {largest_currency['listing_currency']}",
            "Value": largest_currency["share_of_priced_positions_pct"],
            "Unit": "% of priced positions",
            "Basis": "Grouped by instrument listing currency; this is not a look-through measure of underlying currency exposure or fund hedging",
            "Source refs": ", ".join(largest_currency["source_refs"]) or "None",
        })
        packet["data_gaps"].append({
            "item": "Underlying currency exposure",
            "gap": "Instrument listing currency does not reveal the currencies of underlying holdings or whether each fund hedges FX; those details are not available in the sourced data",
        })

    if not positions.empty and priced_total > 0:
        priced_positions["briefing_weight_pct"] = priced_positions["market_value_base"] / priced_total * 100.0
        largest = priced_positions.sort_values("briefing_weight_pct", ascending=False).iloc[0]
        ticker = str(largest["ticker"])
        refs = ["L01", quote_ref_by_ticker.get(ticker, "")]
        fx_ref = fx_ref_by_currency.get(str(largest.get("currency", "")).upper())
        if fx_ref:
            refs.append(fx_ref)
        packet["facts"].append({
            "ID": "F03",
            "Fact": f"Largest instrument position: {ticker}",
            "Value": float(largest["briefing_weight_pct"]),
            "Unit": "% of priced positions",
            "Basis": f"Position value {float(largest['market_value_base']):.2f} {currency} divided by priced-position value",
            "Source refs": ", ".join(ref for ref in refs if ref),
        })

    holdings = issuer_holdings.copy() if isinstance(issuer_holdings, pd.DataFrame) else pd.DataFrame()
    if not holdings.empty:
        holdings["fund_ticker"] = holdings["fund_ticker"].fillna("").astype(str).str.upper()
        # Saved snapshots may include ETFs previously researched but no longer
        # held. Keep sources limited to funds that contribute to this briefing.
        held_tickers = set(priced_positions["ticker"].astype(str).str.upper()) if not priced_positions.empty else set()
        holdings = holdings[holdings["fund_ticker"].isin(held_tickers)].copy()
    if not holdings.empty:
        source_columns = [column for column in ("fund_ticker", "source_name", "source_url", "as_of", "imported_at") if column in holdings]
        distinct_sources = holdings[source_columns].drop_duplicates() if source_columns else pd.DataFrame()
        for record in distinct_sources.to_dict(orient="records"):
            fund = str(record.get("fund_ticker", "")).upper()
            ref = f"H{len(holding_refs_by_fund) + 1:02d}"
            # Keep a stable reference when a fund has multiple source rows.
            if fund in holding_refs_by_fund:
                continue
            holding_refs_by_fund[fund] = [ref]
            add_source(
                ref,
                record.get("source_name", "Issuer holdings snapshot"),
                record.get("as_of"),
                record.get("source_url"),
                f"Fund: {fund}; app import time: {record.get('imported_at') or 'not recorded'}",
            )

    xray = calculate_portfolio_xray(portfolio, holdings)
    coverage = xray["coverage"].copy()
    if not coverage.empty:
        coverage["Source refs"] = coverage["Fund"].astype(str).str.upper().map(
            lambda fund: ", ".join(holding_refs_by_fund.get(fund, [])) or "No issuer snapshot"
        )
        packet["issuer_holdings_coverage"] = coverage.where(pd.notna(coverage), None).to_dict(orient="records")
    xray_coverage_pct = float(xray.get("coverage_pct", 0.0) or 0.0)
    packet["xray_coverage_pct"] = xray_coverage_pct
    packet["facts"].append({
        "ID": "F04",
        "Fact": "Sourced X-Ray holdings coverage",
        "Value": xray_coverage_pct,
        "Unit": "% of priced-position value",
        "Basis": f"Latest saved issuer/imported constituent snapshots; their individual source dates are shown in the source table; priced portfolio value {priced_total:.2f} {currency}",
        "Source refs": ", ".join(dict.fromkeys([
            "L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values(),
            *(ref for refs in holding_refs_by_fund.values() for ref in refs),
        ])) or "None",
    })

    def exposure_records(frame: pd.DataFrame, label_column: str, limit: int = 10) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        output = []
        for row in frame.head(limit).to_dict(orient="records"):
            if "held_through" in row:
                funds = [part.strip().upper() for part in str(row.get("held_through", "")).split(",") if part.strip()]
                refs = list(dict.fromkeys([
                    "L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values(),
                    *(ref for fund in funds for ref in holding_refs_by_fund.get(fund, [])),
                ]))
            else:
                refs = list(dict.fromkeys([
                    "L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values(),
                    *(ref for fund_refs in holding_refs_by_fund.values() for ref in fund_refs),
                ]))
            exposure = {
                "label": _clean(row.get(label_column)),
                "exposure_value_reporting_currency": _clean(row.get("exposure_value")),
                "share_of_priced_positions_pct": _clean(row.get("portfolio_weight_pct")),
                "source_refs": refs,
            }
            if "etf_count" in row:
                exposure["etf_count"] = _clean(row.get("etf_count"))
            if "held_through" in row:
                exposure["held_through"] = _clean(row.get("held_through"))
            if "constituent_ticker" in row:
                exposure["ticker"] = _clean(row.get("constituent_ticker"))
            output.append(exposure)
        return output

    packet["underlying_company_exposures"] = exposure_records(xray["companies"], "constituent_name", 15)

    # The AI briefing packet normally includes only the largest companies to
    # keep it small. For a question naming a company, add exact matches from
    # the full sourced company table so smaller positions (for example ASML at
    # 0.4%) are not incorrectly reported as absent.
    focus_stopwords = {
        "about", "allocation", "allocations", "affect", "and", "are", "as", "asset", "assets",
        "average", "bond", "bonds", "bottom", "by", "cad", "chf", "compare", "comparison", "company",
        "corp", "corporation", "countries", "country", "cny", "currency", "currencies", "dkk", "does",
        "etf", "etfs", "eur", "equities", "equity", "exposure", "exposures", "for", "from", "fund",
        "funds", "fx", "gbp", "get", "give", "group", "holding", "holdings", "how", "hkd", "impact",
        "in", "index", "indices", "india", "inr", "inc", "is", "it", "its", "jpy", "largest", "less",
        "limited", "ltd", "market", "markets", "me", "mex", "mxn", "more", "my", "nok", "nzd", "of",
        "on", "or", "performance", "plc", "pln", "portfolio", "position", "positions", "price", "prices",
        "region", "regions", "return", "returns", "risk", "risks", "sa", "sector", "sectors", "se", "share",
        "shares", "show", "stock", "stocks", "tell", "the", "their", "this", "top", "to", "try", "usd",
        "us", "value", "what", "which", "with", "would", "yield", "you", "your", "zar", "adr",
    }
    # Do not confuse a country, sector, region, or asset-class question with
    # a company-name lookup when a label appears in the user's question.
    for dimension, label_column in (
        ("countries", "category"),
        ("sectors", "category"),
        ("regions", "category"),
        ("assets", "Underlying asset"),
    ):
        labels = xray.get(dimension)
        if isinstance(labels, pd.DataFrame) and label_column in labels:
            for label in labels[label_column].dropna().astype(str):
                focus_stopwords.update(token.casefold() for token in re.findall(r"[A-Za-z0-9]+", label))
    query_tokens = re.findall(r"[A-Za-z0-9]+", packet["focus_company_query"])
    focus_terms = {
        token.casefold() for token in query_tokens
        if len(token) >= 3 and token.casefold() not in focus_stopwords
    }
    # Keep short all-caps symbols such as GM as exact search terms too.
    focus_terms.update(
        token.casefold() for token in query_tokens
        if len(token) >= 2 and token.isalpha() and token.isupper()
        and token.casefold() not in focus_stopwords
    )
    company_frame = xray["companies"]
    if focus_terms and not company_frame.empty:
        matched_rows = []
        for company_row in company_frame.to_dict(orient="records"):
            company_tokens = {
                token.casefold()
                for token in re.findall(
                    r"[A-Za-z0-9]+",
                    f"{company_row.get('constituent_name', '')} {company_row.get('constituent_ticker', '')}",
                )
            }
            if focus_terms.intersection(company_tokens):
                matched_rows.append(company_row)
        for match_index, exposure in enumerate(
            exposure_records(pd.DataFrame(matched_rows), "constituent_name", len(matched_rows)),
            start=1,
        ):
            exposure_value = float(exposure.get("exposure_value_reporting_currency") or 0.0)
            share_pct = float(exposure.get("share_of_priced_positions_pct") or 0.0)
            funds = [fund.strip().upper() for fund in str(exposure.get("held_through", "")).split(",") if fund.strip()]
            fund_coverage = coverage[coverage["Fund"].astype(str).str.upper().isin(funds)] if not coverage.empty else pd.DataFrame()
            exposure["issuer_snapshot_dates"] = [
                {"fund": str(row.get("Fund", "")), "as_of": str(row.get("As of", "—")), "source": str(row.get("Source refs", "—"))}
                for row in fund_coverage.to_dict(orient="records")
            ]
            exposure["xray_coverage_pct"] = xray_coverage_pct
            exposure["mechanical_effect_if_company_falls_20_pct_reporting_currency"] = exposure_value * -0.20
            exposure["mechanical_effect_if_company_falls_20_pct_of_portfolio"] = share_pct * -0.20
            exposure["mechanical_effect_assumptions"] = (
                "Illustration only: this constituent's market value falls 20%; all other holdings and FX stay unchanged. "
                "No tracking differences, fees, taxes, cash flows or second-order effects are included."
            )
            exposure["fact_ref"] = f"Q{match_index:02d}"
            packet["facts"].append({
                "ID": exposure["fact_ref"],
                "Fact": f"Sourced look-through exposure: {exposure.get('label', '')}",
                "Value": share_pct,
                "Unit": "% of priced positions",
                "Basis": (
                    f"Issuer-reported constituent holdings represent {exposure_value:.2f} {currency} "
                    f"of value, held through {', '.join(funds) or 'not identified'}; overall X-Ray coverage is {xray_coverage_pct:.2f}%"
                ),
                "Source refs": ", ".join(exposure.get("source_refs", [])) or "None",
            })
            packet["focus_company_exposures"].append(exposure)
    packet["focus_company_lookup"] = {
        "attempted": bool(focus_terms),
        "terms": sorted(focus_terms),
        "match_count": len(packet["focus_company_exposures"]),
    }
    packet["country_exposures"] = exposure_records(xray["countries"], "category", 10)
    packet["sector_exposures"] = exposure_records(xray["sectors"], "category", 10)
    packet["region_exposures"] = exposure_records(xray["regions"], "category", 10)
    packet["other_asset_exposures"] = exposure_records(xray["assets"], "Underlying asset", 10)

    for fact_id, fact_name, records in (
        ("F05", "Largest sourced underlying company exposure", packet["underlying_company_exposures"]),
        ("F06", "Largest sourced country exposure", packet["country_exposures"]),
        ("F07", "Largest sourced sector exposure", packet["sector_exposures"]),
    ):
        if records:
            largest_exposure = records[0]
            packet["facts"].append({
                "ID": fact_id,
                "Fact": f"{fact_name}: {largest_exposure['label']}",
                "Value": largest_exposure["share_of_priced_positions_pct"],
                "Unit": "% of priced-position value",
                "Basis": f"Sourced look-through value {largest_exposure['exposure_value_reporting_currency']} {currency}; not classified/uncovered positions are excluded",
                "Source refs": ", ".join(largest_exposure["source_refs"]) or "None",
            })

    if not positions.empty and priced_total > 0:
        largest = priced_positions.sort_values("market_value_base", ascending=False).iloc[0]
        packet["potential_risks"].append({
            "type": "largest_instrument",
            "ticker": str(largest["ticker"]),
            "share_pct": float(largest["market_value_base"]) / priced_total * 100.0,
            "ref": "F03",
        })
    if packet["underlying_company_exposures"]:
        company = packet["underlying_company_exposures"][0]
        packet["potential_risks"].append({"type": "largest_company", **company, "ref": "F05"})
    if packet["country_exposures"]:
        country_exposure = packet["country_exposures"][0]
        packet["potential_risks"].append({"type": "largest_country", **country_exposure, "ref": "F06"})
    if packet["sector_exposures"]:
        sector_exposure = packet["sector_exposures"][0]
        packet["potential_risks"].append({"type": "largest_sector", **sector_exposure, "ref": "F07"})
    if packet["listing_currency_exposures"]:
        listing_currency = packet["listing_currency_exposures"][0]
        packet["potential_risks"].append({"type": "largest_listing_currency", **listing_currency, "ref": "F08"})
    if xray_coverage_pct < 100.0:
        packet["potential_risks"].append({
            "type": "lookthrough_coverage",
            "share_pct": xray_coverage_pct,
            "ref": "F04",
        })
    if not complete and not positions.empty:
        packet["potential_risks"].append({"type": "incomplete_valuation", "ref": "F02"})
    if priced_total > 0 and not packet["country_exposures"]:
        packet["data_gaps"].append({
            "item": "Country classifications",
            "gap": "No dated country classifications are available, so country concentration and a country-market stress scenario cannot be measured",
        })

    if not xray["countries"].empty and priced_total > 0:
        country = xray["countries"].iloc[0]
        exposure_value = float(country["exposure_value"])
        country_share = exposure_value / priced_total * 100.0
        shock_pct = -20.0
        country_name = str(country["category"])
        direct_impact = exposure_value * shock_pct / 100.0
        packet["stress_scenario"] = {
            "ID": "S01",
            "scenario_type": "country",
            "market": country_name,
            "sourced_exposure_value_reporting_currency": exposure_value,
            "sourced_exposure_share_of_priced_positions_pct": country_share,
            "assumed_market_change_pct": shock_pct,
            "mechanical_portfolio_value_change_reporting_currency": direct_impact,
            "mechanical_change_as_share_of_priced_positions_pct": direct_impact / priced_total * 100.0,
            "source_refs": list(dict.fromkeys([
                "L01", *quote_ref_by_ticker.values(), *fx_ref_by_currency.values(),
                *country.get("source_refs", []),
                "F06",
            ])),
            "assumptions": [
                f"Only the sourced constituent exposure classified as {country_name} falls by 20%.",
                "All other prices and FX rates stay unchanged.",
                "No cross-market effects, fund tracking differences, fees, tax, cash flows, or second-order effects are included.",
                "This is a mechanical stress calculation, not a forecast; unclassified and uncovered exposures are not included in the shocked amount.",
            ],
        }
    elif priced_total > 0:
        if not priced_positions.empty:
            largest = priced_positions.sort_values("market_value_base", ascending=False).iloc[0]
            exposure_value = float(largest["market_value_base"])
            impact = exposure_value * -20.0 / 100.0
            ticker = str(largest["ticker"])
            ticker_refs = ["L01", quote_ref_by_ticker.get(ticker, "")]
            fx_ref = fx_ref_by_currency.get(str(largest.get("currency", "")).upper())
            if fx_ref:
                ticker_refs.append(fx_ref)
            ticker_refs.append("F03")
            packet["stress_scenario"] = {
                "ID": "S02",
                "scenario_type": "instrument",
                "market": ticker,
                "sourced_exposure_value_reporting_currency": exposure_value,
                "sourced_exposure_share_of_priced_positions_pct": exposure_value / priced_total * 100.0,
                "assumed_market_change_pct": -20.0,
                "mechanical_portfolio_value_change_reporting_currency": impact,
                "mechanical_change_as_share_of_priced_positions_pct": impact / priced_total * 100.0,
                "source_refs": list(dict.fromkeys(ref for ref in ticker_refs if ref)),
                "assumptions": [
                    f"The market price of {ticker} falls by 20%.",
                    "All other instrument prices and FX rates stay unchanged.",
                    "No fees, taxes, cash flows, tracking differences, or second-order effects are included.",
                    "This is a mechanical stress calculation, not a forecast.",
                ],
            }

    if xray_coverage_pct < 100.0:
        packet["data_gaps"].append({
            "item": "Underlying holdings",
            "gap": f"Issuer snapshots cover {xray_coverage_pct:.2f}% of priced-position value; the remainder is not assigned to companies or country/sector exposures",
        })
    country_coverage = float(pd.to_numeric(xray["countries"].get("exposure_value", pd.Series(dtype=float)), errors="coerce").sum()) if not xray["countries"].empty else 0.0
    country_coverage_pct = country_coverage / priced_total * 100.0 if priced_total > 0 else 0.0
    if not xray["countries"].empty and country_coverage_pct + 0.01 < xray_coverage_pct:
        packet["data_gaps"].append({
            "item": "Country classifications",
            "gap": f"Country-classified weight is {country_coverage_pct:.2f}% of priced-position value, below total X-Ray holdings coverage of {xray_coverage_pct:.2f}%",
        })

    packet["sources"] = source_rows
    packet["data_quality"] = {
        "priced_positions": int(positions["market_value_base"].notna().sum()) if not positions.empty else 0,
        "total_positions": int(len(positions)),
        "priced_value_reporting_currency": priced_total,
        "valuation_complete": complete,
        "xray_coverage_pct": xray_coverage_pct,
        "country_classification_coverage_pct": country_coverage_pct,
    }
    packet["facts"].sort(key=lambda fact: int(str(fact.get("ID", "F0"))[1:]))
    return packet
