import unittest

import numpy as np
import pandas as pd

from performance_engine import (
    compare_to_benchmark,
    convert_prices_to_currency,
    linked_attribution,
    performance_summary,
    portfolio_total_return,
)


class PerformanceEngineTests(unittest.TestCase):
    def test_buy_does_not_appear_as_investment_profit(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        prices = pd.DataFrame({"ABC": [100.0, 110.0, 121.0]}, index=dates)
        transactions = pd.DataFrame([
            {"trade_date": "2023-12-31", "ticker": "ABC", "transaction_type": "OPENING", "quantity": 1},
            {"trade_date": "2024-01-02", "ticker": "ABC", "transaction_type": "BUY", "quantity": 10},
        ])
        result = portfolio_total_return(prices, transactions)
        self.assertAlmostEqual(result.loc[dates[1], "daily_return"], 0.10, places=6)
        self.assertAlmostEqual(result.loc[dates[2], "daily_return"], 0.10, places=6)
        self.assertAlmostEqual(result.loc[dates[2], "portfolio_index"], 121.0, places=4)

    def test_reporting_currency_includes_fx_movement(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
        prices = pd.DataFrame({"ABC": [100.0, 110.0]}, index=dates)
        fx = pd.Series([4.0, 4.2], index=dates)
        converted = convert_prices_to_currency(prices, {"ABC": "EUR"}, {"EUR": fx}, "PLN")
        self.assertAlmostEqual(converted.loc[dates[0], "ABC"], 400.0)
        self.assertAlmostEqual(converted.loc[dates[1], "ABC"], 462.0)
        self.assertAlmostEqual(converted.loc[dates[1], "ABC"] / converted.loc[dates[0], "ABC"] - 1, 0.155)

    def test_carino_attribution_sums_to_linked_portfolio_return(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        prices = pd.DataFrame({"AAA": [100.0, 110.0, 105.0], "BBB": [100.0, 100.0, 110.0]}, index=dates)
        transactions = pd.DataFrame([
            {"trade_date": "2023-12-31", "ticker": "AAA", "transaction_type": "OPENING", "quantity": 1},
            {"trade_date": "2023-12-31", "ticker": "BBB", "transaction_type": "OPENING", "quantity": 1},
        ])
        result = portfolio_total_return(prices, transactions)
        attribution = linked_attribution(result)
        valid_returns = result["daily_return"].dropna()
        total = (1 + valid_returns).prod() - 1
        self.assertAlmostEqual(attribution["Contribution %"].sum() / 100, total, places=7)

    def test_benchmark_comparison_uses_common_dates(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        portfolio = pd.DataFrame({"daily_return": [np.nan, 0.10, 0.05]}, index=dates)
        benchmark = pd.Series([100.0, 105.0, 110.25], index=dates)
        comparison = compare_to_benchmark(portfolio, benchmark)
        self.assertEqual(len(comparison), 2)
        self.assertAlmostEqual(comparison["Portfolio index"].iloc[-1] / comparison["Portfolio index"].iloc[0] - 1, 0.05)
        self.assertAlmostEqual(performance_summary(portfolio)["total_return"], 0.155)

    def test_missing_ticker_history_is_not_silently_excluded(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        prices = pd.DataFrame({"AAA": [100.0, 101.0, 102.0]}, index=dates)
        transactions = pd.DataFrame([
            {"trade_date": "2023-12-31", "ticker": "AAA", "transaction_type": "OPENING", "quantity": 1},
            {"trade_date": "2023-12-31", "ticker": "MISSING", "transaction_type": "OPENING", "quantity": 1},
        ])
        result = portfolio_total_return(prices, transactions)
        self.assertTrue(pd.isna(result.loc[dates[1], "daily_return"]))

    def test_missing_held_price_is_not_silently_zero_return(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        prices = pd.DataFrame({"AAA": [100.0, np.nan, 110.0]}, index=dates)
        transactions = pd.DataFrame([{"trade_date": "2023-12-31", "ticker": "AAA", "transaction_type": "OPENING", "quantity": 1}])
        result = portfolio_total_return(prices, transactions, max_stale_rows=0)
        self.assertTrue(pd.isna(result.loc[dates[1], "daily_return"]))


if __name__ == "__main__":
    unittest.main()
