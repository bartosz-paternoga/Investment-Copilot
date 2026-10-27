import unittest
import pandas as pd
from advanced_analytics import historical_risk_table, portfolio_xirr, xirr


class AdvancedAnalyticsTests(unittest.TestCase):
    def test_xirr_known_annual_return(self):
        result = xirr([("2023-01-01", -1000), ("2024-01-01", 1100)])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.10, places=4)

    def test_xirr_requires_both_signs(self):
        self.assertIsNone(xirr([("2024-01-01", -1000), ("2025-01-01", -100)]))

    def test_portfolio_xirr_uses_investment_flows_and_fx(self):
        tx = pd.DataFrame([
            {"trade_date":"2024-01-01", "transaction_type":"OPENING", "quantity":10, "unit_price":100, "amount":0, "fee":0, "currency":"EUR"},
            {"trade_date":"2025-01-01", "transaction_type":"DIVIDEND", "quantity":0, "unit_price":0, "amount":20, "fee":0, "currency":"EUR"},
        ])
        result = portfolio_xirr(tx, 1100, "2025-01-01", "PLN", lambda source, target, on_date: 1.0)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)

    def test_risk_table_ignores_insufficient_series(self):
        history = pd.DataFrame({"A":[100, 101, 99, 102], "B":[100, None, None, None]})
        result = historical_risk_table(history)
        self.assertEqual(result["Ticker"].tolist(), ["A"])
        self.assertIn("Maximum drawdown %", result.columns)


if __name__ == "__main__":
    unittest.main()
