"""Fetch sourced ETF constituent snapshots from public issuer holdings pages.

The optional scraper adapters read public Amundi, iShares, Vanguard and
Xtrackers listings/holdings. No portfolio values are sent to those providers.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
import re
from typing import Any

import pandas as pd


# These two common listings are also a fallback if Yahoo does not return ISIN.
# They identify the share class, not a holding inside the fund.
KNOWN_FUND_ISINS = {
    "EXUS.DE": "IE0006WW1TQ4",
    "VWCE.DE": "IE00BK5BQT80",
}

# Route the two known European listings straight to their issuer. Apart from
# being faster, this avoids downloading three unrelated issuer catalogues
# before reaching Xtrackers or Vanguard on every first load.
KNOWN_ISIN_ISSUERS = {
    "IE0006WW1TQ4": "Xtrackers",
    "IE00BK5BQT80": "Vanguard",
}

ISSUER_HOME = {
    "Amundi": "https://www.amundietf.com/",
    "iShares": "https://www.ishares.com/",
    "Vanguard": "https://global.vanguard.com/",
    "Xtrackers": "https://etf.dws.com/",
}


def _resolved_isin(ticker: str, supplied_isin: str = "") -> str:
    isin = str(supplied_isin or "").strip().upper()
    if isin:
        return isin
    ticker = str(ticker or "").strip().upper()
    if ticker in KNOWN_FUND_ISINS:
        return KNOWN_FUND_ISINS[ticker]
    try:
        import yfinance as yf

        instrument = yf.Ticker(ticker)
        isin = str(getattr(instrument, "isin", "") or "").strip().upper()
        if isin:
            return isin
        info = instrument.info or {}
        return str(info.get("isin", "") or "").strip().upper()
    except Exception:
        return ""


def _source_classes() -> list[tuple[str, Any]]:
    try:
        from portfolio_scraper.etf import (
            AmundiScraper,
            ISharesItScraper,
            VanguardItScraper,
            XTrackersItScraper,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The issuer holdings connector is not installed. Restart the app so its dependencies can finish installing."
        ) from exc
    return [
        ("Amundi", AmundiScraper),
        ("iShares", ISharesItScraper),
        ("Vanguard", VanguardItScraper),
        ("Xtrackers", XTrackersItScraper),
    ]


@lru_cache(maxsize=4)
def _issuer_listings(issuer: str, scraper_class: Any) -> pd.DataFrame:
    """Reuse issuer product catalogs while separately refreshing fund holdings."""
    listings = scraper_class().get_listings()
    return listings.copy() if isinstance(listings, pd.DataFrame) else pd.DataFrame()


def _pick_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    by_normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): str(c) for c in frame.columns}
    for name in names:
        match = by_normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if match is not None:
            return match
    return None


def _source_as_of(frame: pd.DataFrame) -> str:
    date_column = _pick_column(
        frame,
        ("as_of", "asof", "as_of_date", "holdings_date", "portfolio_date", "report_date", "date"),
    )
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce", utc=True).dropna()
        if not dates.empty:
            return dates.max().date().isoformat()
    return date.today().isoformat()


def _source_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    # The adapter's normalized sector fields can be Enum values.
    return str(getattr(value, "value", value)).strip()


def fetch_issuer_holdings(fund_ticker: str, isin: str = "") -> dict[str, Any]:
    """Fetch one ETF snapshot; return normalized data plus source/date metadata."""
    fund = str(fund_ticker or "").strip().upper()
    resolved_isin = _resolved_isin(fund, isin)
    if not resolved_isin:
        raise ValueError(
            f"Could not identify {fund} by ISIN. Add its ISIN in ETF Research → Instrument Master."
        )

    all_sources = _source_classes()
    preferred_issuer = KNOWN_ISIN_ISSUERS.get(resolved_isin)
    # For a known ISIN use the matching provider only. For other funds, search
    # the supported providers in turn, using their catalog as the identifier
    # lookup.
    sources = (
        [source for source in all_sources if source[0] == preferred_issuer]
        if preferred_issuer
        else all_sources
    )
    failures = []
    for issuer, scraper_class in sources:
        try:
            scraper = scraper_class()
            listings = pd.DataFrame()
            listing = pd.Series(dtype=object)
            # Xtrackers' documented holdings identifier is the ISIN itself;
            # do not make an unnecessary catalog request for that provider.
            if issuer == "Xtrackers":
                internal_id = resolved_isin
            else:
                listings = _issuer_listings(issuer, scraper_class)
                isin_column = _pick_column(listings, ("isin",)) if not listings.empty else None
                if isin_column is None:
                    if preferred_issuer:
                        raise ValueError("the issuer product list did not include an ISIN column")
                    continue
                matched = listings[
                    listings[isin_column].fillna("").astype(str).str.strip().str.upper() == resolved_isin
                ]
                if matched.empty:
                    if preferred_issuer:
                        raise ValueError("the issuer product list did not include this fund ISIN")
                    continue
                listing = matched.iloc[0]
                internal_id = listing.get("internal_id", resolved_isin)
                if pd.isna(internal_id) or not str(internal_id).strip():
                    internal_id = resolved_isin

            # Use the scraper's normalized holdings response. Fetching both its
            # raw and normalized forms made the same issuer file download
            # twice, slowing first page load and increasing timeout risk.
            normalized = scraper.get_holdings(internal_id)
            if normalized is None or normalized.empty:
                raise ValueError("the issuer returned no holdings rows")

            name_col = _pick_column(normalized, ("name", "holding_name", "security_name"))
            ticker_col = _pick_column(normalized, ("ticker", "symbol"))
            isin_col = _pick_column(normalized, ("isin",))
            weight_col = _pick_column(normalized, ("weight", "weight_pct", "holding_percent"))
            sector_col = _pick_column(normalized, ("sector",))
            country_col = _pick_column(normalized, ("country",))
            region_col = _pick_column(normalized, ("region",))
            if weight_col is None:
                raise ValueError("the issuer response did not include holding weights")

            rows = pd.DataFrame(index=normalized.index)
            names = normalized[name_col] if name_col else pd.Series("", index=normalized.index)
            tickers = normalized[ticker_col] if ticker_col else pd.Series("", index=normalized.index)
            holding_isins = normalized[isin_col] if isin_col else pd.Series("", index=normalized.index)
            rows["constituent_name"] = names.fillna("").astype(str).str.strip()
            rows["constituent_ticker"] = tickers.fillna("").astype(str).str.strip().str.upper()
            missing_id = rows["constituent_ticker"].eq("")
            rows.loc[missing_id, "constituent_ticker"] = holding_isins[missing_id].fillna("").astype(str).str.strip().str.upper()
            missing_id = rows["constituent_ticker"].eq("")
            rows.loc[missing_id, "constituent_ticker"] = rows.loc[missing_id, "constituent_name"].str.upper()
            rows["weight_pct"] = pd.to_numeric(normalized[weight_col], errors="coerce") * 100.0
            rows["sector"] = normalized[sector_col].map(_source_text) if sector_col else ""
            rows["country"] = normalized[country_col].map(_source_text) if country_col else ""
            rows["region"] = normalized[region_col].map(_source_text) if region_col else ""
            rows = rows.dropna(subset=["weight_pct"])
            rows = rows[(rows["constituent_ticker"] != "") & (rows["weight_pct"] >= 0)]
            if rows.empty:
                raise ValueError("the issuer response contained no usable constituent rows")

            # Providers can report multiple share lines for one company. Keep
            # their total weight while retaining the first available labels.
            rows = rows.groupby("constituent_ticker", as_index=False, sort=False).agg(
                constituent_name=("constituent_name", lambda values: next((v for v in values if v), "")),
                weight_pct=("weight_pct", "sum"),
                sector=("sector", lambda values: next((v for v in values if v), "")),
                country=("country", lambda values: next((v for v in values if v), "")),
                region=("region", lambda values: next((v for v in values if v), "")),
            )
            source_date = _source_as_of(normalized)
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows["_source_date"] = source_date
            url_column = _pick_column(listings, ("product_url", "fund_url", "url", "website")) if not listings.empty else None
            url_value = listing.get(url_column, "") if url_column else ""
            source_url = "" if pd.isna(url_value) else str(url_value).strip()
            if not source_url.startswith(("https://", "http://")):
                source_url = ISSUER_HOME.get(issuer, "")
            return {
                "fund_ticker": fund,
                "fund_isin": resolved_isin,
                "holdings": rows,
                "source_name": f"{issuer} issuer holdings · ISIN {resolved_isin}",
                "source_url": source_url,
                "as_of": source_date,
                "fetched_at": fetched_at,
                "issuer": issuer,
                "holding_count": int(len(rows)),
            }
        except Exception as exc:
            failures.append(f"{issuer}: {exc}")

    detail = "; ".join(failures[-3:])
    raise ValueError(
        f"Could not load issuer holdings for {fund} (ISIN {resolved_isin})."
        + (f" Source details: {detail}" if detail else "")
    )
