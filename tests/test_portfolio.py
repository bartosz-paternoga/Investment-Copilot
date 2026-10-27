import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import portfolio


class LedgerAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_db = portfolio.DB_PATH
        self.old_legacy = portfolio.LEGACY_HOLDINGS_PATH
        portfolio.DB_PATH = Path(self.temp.name) / "portfolio.sqlite3"
        portfolio.LEGACY_HOLDINGS_PATH = Path(self.temp.name) / "holdings.csv"
        portfolio.FX_CACHE.clear()
        portfolio.initialize_db()

    def tearDown(self):
        portfolio.DB_PATH = self.old_db
        portfolio.LEGACY_HOLDINGS_PATH = self.old_legacy
        self.temp.cleanup()

    def test_weighted_average_cost_realized_gain_and_dividends(self):
        df = pd.DataFrame([
            {"id": 1, "trade_date": "2025-01-02", "ticker": "ABC", "transaction_type": "BUY", "quantity": 10, "unit_price": 100, "currency": "PLN", "amount": 0, "fee": 5},
            {"id": 2, "trade_date": "2025-03-02", "ticker": "ABC", "transaction_type": "SELL", "quantity": 4, "unit_price": 120, "currency": "PLN", "amount": 0, "fee": 2},
            {"id": 3, "trade_date": "2025-04-02", "ticker": "ABC", "transaction_type": "DIVIDEND", "quantity": 0, "unit_price": 0, "currency": "PLN", "amount": 20, "fee": 0},
        ])
        with patch.object(portfolio, "fx_rate", return_value=1.0):
            summary = portfolio.transaction_types_summary(df, "PLN")
        self.assertAlmostEqual(summary["positions"]["ABC"]["quantity"], 6)
        self.assertAlmostEqual(summary["positions"]["ABC"]["cost_basis_base"], 603)
        self.assertAlmostEqual(summary["realized_pl"], 76)
        self.assertAlmostEqual(summary["dividends"], 20)
        self.assertAlmostEqual(summary["fees"], 7)

    def test_oversell_is_rejected_and_not_saved(self):
        portfolio.add_transaction({"trade_date":"2026-01-01", "ticker":"ABC", "transaction_type":"OPENING",
                                   "quantity":2, "unit_price":50, "currency":"PLN"})
        with self.assertRaises(ValueError):
            portfolio.add_transaction({"trade_date":"2026-01-02", "ticker":"ABC", "transaction_type":"SELL",
                                       "quantity":3, "unit_price":60, "currency":"PLN"})
        tx = portfolio.list_transactions()
        self.assertEqual(len(tx), 1)
        self.assertEqual(tx.iloc[0]["transaction_type"], "OPENING")

    def test_opening_unknown_cost_marks_cost_basis_unknown(self):
        df = pd.DataFrame([{"id":1,"trade_date":"2026-01-01","ticker":"ABC","transaction_type":"OPENING",
                            "quantity":3,"unit_price":0,"currency":"PLN","amount":0,"fee":0}])
        with patch.object(portfolio, "fx_rate", return_value=1.0):
            summary = portfolio.transaction_types_summary(df, "PLN")
        self.assertFalse(summary["positions"]["ABC"]["cost_basis_known"])
        self.assertIsNone(summary["positions"]["ABC"]["cost_basis_base"])


    def test_position_valuation_and_unrealized_pl(self):
        transactions = pd.DataFrame([
            {"id": 1, "trade_date": "2026-01-02", "ticker": "ABC", "transaction_type": "BUY", "quantity": 10, "unit_price": 100, "currency": "PLN", "amount": 0, "fee": 0}
        ])
        quotes = pd.DataFrame([{"ticker":"ABC", "name":"Example ETF", "currency":"PLN", "price":120, "previous_close":119,
                               "day_change_pct":100*(120/119-1), "price_date":"2026-01-03", "fetched_at":"2026-01-03T15:00:00+00:00",
                               "source":"test", "status":"OK", "error":""}])
        with patch.object(portfolio, "fx_rate", return_value=1.0):
            report = portfolio.build_portfolio(transactions, quotes, "PLN")
        self.assertEqual(report.iloc[0]["quantity"], 10)
        self.assertAlmostEqual(report.iloc[0]["market_value_base"], 1200)
        self.assertAlmostEqual(report.iloc[0]["unrealized_pl_base"], 200)
        metrics = portfolio.portfolio_metrics(report, "PLN")
        self.assertAlmostEqual(metrics["unrealized_pl"], 200)

    def test_deleting_buy_that_underpins_sale_is_rolled_back(self):
        opening_id = portfolio.add_transaction({"trade_date":"2026-01-01", "ticker":"ABC", "transaction_type":"OPENING",
                                                 "quantity":5, "unit_price":10, "currency":"PLN"})
        portfolio.add_transaction({"trade_date":"2026-01-02", "ticker":"ABC", "transaction_type":"SELL",
                                   "quantity":3, "unit_price":12, "currency":"PLN"})
        with self.assertRaises(ValueError):
            portfolio.delete_transaction(opening_id)
        self.assertEqual(len(portfolio.list_transactions()), 2)

    def test_simulation_returns_requested_horizons_and_nonnegative_values(self):
        out = portfolio.simulate_forecast(10000, 100, 0.05, 0.15, 0.025, simulations=500, seed=1, reporting_currency="EUR")
        self.assertEqual(out["Horizon (years)"].tolist(), [1, 3, 5, 10])
        self.assertTrue((out["Median (nominal)"] >= 0).all())
        self.assertIn("Median (today's EUR)", out.columns)

    def test_instrument_metadata_persists_custom_display_name(self):
        portfolio.save_instrument("ABC", display_name="My custom ETF", isin="IE00TEST1234", exchange="XETRA")
        instruments = portfolio.list_instruments()
        row = instruments.loc[instruments["ticker"] == "ABC"].iloc[0]
        self.assertEqual(row["display_name"], "My custom ETF")
        self.assertEqual(row["exchange"], "XETRA")

    def test_quick_position_can_be_added_edited_and_removed(self):
        status = portfolio.replace_quick_position("ABC", 12, 25.5, "EUR", "2026-01-02", "My ETF")
        self.assertEqual(status, "added")
        tx = portfolio.list_transactions()
        self.assertEqual(len(tx), 1)
        self.assertEqual(float(tx.iloc[0]["quantity"]), 12)
        status = portfolio.replace_quick_position("ABC", 15, 26, "EUR", "2026-02-02", "My ETF")
        self.assertEqual(status, "updated")
        tx = portfolio.list_transactions()
        self.assertEqual(len(tx), 1)
        self.assertEqual(float(tx.iloc[0]["quantity"]), 15)
        status = portfolio.replace_quick_position("ABC", 0, 26, "EUR", "2026-02-02")
        self.assertEqual(status, "removed")
        self.assertTrue(portfolio.list_transactions().empty)

    def test_quick_position_edit_protects_real_trade_history(self):
        portfolio.add_transaction({"trade_date":"2026-01-01", "ticker":"ABC", "transaction_type":"OPENING",
                                  "quantity":5, "unit_price":10, "currency":"EUR"})
        portfolio.add_transaction({"trade_date":"2026-02-01", "ticker":"ABC", "transaction_type":"BUY",
                                  "quantity":1, "unit_price":11, "currency":"EUR"})
        with self.assertRaises(ValueError):
            portfolio.replace_quick_position("ABC", 7, 10, "EUR", "2026-03-01")
        self.assertEqual(len(portfolio.list_transactions()), 2)

    def test_trade_currency_prefers_valid_quote_metadata(self):
        quotes = pd.DataFrame([{"ticker":"ABC", "currency":"PLN"}])
        transactions = pd.DataFrame([{"id":1, "trade_date":"2026-01-01", "ticker":"ABC", "currency":"EUR"}])
        self.assertEqual(portfolio.resolve_position_currency("abc", quotes, transactions, "USD"), "PLN")

    def test_trade_currency_falls_back_to_latest_ledger_currency_when_quote_is_missing(self):
        quotes = pd.DataFrame([{"ticker":"ABC", "currency":""}])
        transactions = pd.DataFrame([
            {"id":1, "trade_date":"2026-01-01", "ticker":"ABC", "currency":"EUR"},
            {"id":2, "trade_date":"2026-03-01", "ticker":"ABC", "currency":"USD"},
        ])
        self.assertEqual(portfolio.resolve_position_currency("ABC", quotes, transactions, "PLN"), "USD")

    def test_trade_currency_uses_reporting_fallback_only_without_quote_or_ledger_currency(self):
        self.assertEqual(portfolio.resolve_position_currency("ABC", pd.DataFrame(), pd.DataFrame(), "GBP"), "GBP")

    def test_remove_ticker_removes_all_its_ledger_rows(self):
        portfolio.add_transaction({"trade_date":"2026-01-01", "ticker":"ABC", "transaction_type":"OPENING",
                                  "quantity":5, "unit_price":10, "currency":"EUR"})
        portfolio.add_transaction({"trade_date":"2026-01-01", "ticker":"XYZ", "transaction_type":"OPENING",
                                  "quantity":2, "unit_price":20, "currency":"USD"})
        deleted = portfolio.remove_ticker_from_portfolio("ABC")
        self.assertEqual(deleted, 1)
        self.assertEqual(portfolio.list_transactions()["ticker"].tolist(), ["XYZ"])

    def test_watchlist_add_and_remove(self):
        portfolio.add_watchlist_ticker("ABC", "research later")
        self.assertEqual(portfolio.list_watchlist().iloc[0]["ticker"], "ABC")
        portfolio.remove_watchlist_ticker("ABC")
        self.assertTrue(portfolio.list_watchlist().empty)

    def test_sourced_etf_constituent_snapshot_persists_provenance(self):
        count = portfolio.save_etf_holdings("FUND.A", pd.DataFrame([
            {"constituent_ticker": "AAPL", "constituent_name": "Apple Inc.", "weight_pct": 6.5},
            {"constituent_ticker": "MSFT", "constituent_name": "Microsoft", "weight_pct": 5.5},
        ]), source_name="Example issuer factsheet", source_url="https://example.com/factsheet", as_of="2026-09-30")
        self.assertEqual(count, 2)
        held = portfolio.list_etf_holdings()
        self.assertEqual(held["fund_ticker"].unique().tolist(), ["FUND.A"])
        self.assertTrue((held["source_name"] == "Example issuer factsheet").all())
        self.assertTrue((held["as_of"] == "2026-09-30").all())

    def test_sourced_etf_holdings_reject_duplicate_constituents(self):
        duplicate = pd.DataFrame([
            {"constituent_ticker": "AAPL", "weight_pct": 5},
            {"constituent_ticker": "AAPL", "weight_pct": 4},
        ])
        with self.assertRaises(ValueError):
            portfolio.save_etf_holdings("FUND.A", duplicate, source_name="Source", as_of="2026-09-30")

    def test_quote_cache_serves_recent_quote_without_provider_call(self):
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with portfolio._connect() as con:
            con.execute("""INSERT INTO quote_cache(ticker,name,currency,price,previous_close,day_change_pct,price_date,
                fetched_at,source,status,error) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                ("ABC", "Example ETF", "EUR", 101.25, 100.0, 1.25, datetime.now().date().isoformat(),
                 fetched_at, "test source", "OK", ""))
        with patch.object(portfolio, "_fetch_quotes_uncached") as provider:
            result = portfolio.get_quotes(["ABC"])
        provider.assert_not_called()
        self.assertAlmostEqual(float(result.iloc[0]["price"]), 101.25)
        self.assertEqual(result.iloc[0]["source"], "test source")

    def test_history_cache_can_be_read_without_provider_dependency(self):
        now = datetime.now().date()
        with portfolio._connect() as con:
            con.executemany("INSERT INTO market_prices(ticker,price_date,adjusted_close,fetched_at) VALUES(?,?,?,?)",
                            [("ABC", (now - timedelta(days=2)).isoformat(), 100.0, "test"),
                             ("ABC", (now - timedelta(days=1)).isoformat(), 102.0, "test")])
        with patch.object(portfolio, "yf", None):
            history = portfolio.get_history(["ABC"], "1y")
        self.assertEqual(history["ABC"].dropna().tolist(), [100.0, 102.0])


if __name__ == "__main__":
    unittest.main()
