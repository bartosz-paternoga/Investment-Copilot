import unittest
import pandas as pd

from research_analytics import calculate_etf_overlap, holdings_weight_coverage


class ResearchAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.holdings = pd.DataFrame([
            {"fund_ticker": "AAA", "constituent_ticker": "AAPL", "weight_pct": 5.0, "as_of": "2026-09-30", "source_name": "Issuer A", "source_url": "https://example.com/a"},
            {"fund_ticker": "AAA", "constituent_ticker": "MSFT", "weight_pct": 3.0, "as_of": "2026-09-30", "source_name": "Issuer A", "source_url": "https://example.com/a"},
            {"fund_ticker": "BBB", "constituent_ticker": "AAPL", "weight_pct": 4.0, "as_of": "2026-09-30", "source_name": "Issuer B", "source_url": "https://example.com/b"},
            {"fund_ticker": "BBB", "constituent_ticker": "MSFT", "weight_pct": 7.0, "as_of": "2026-09-30", "source_name": "Issuer B", "source_url": "https://example.com/b"},
            {"fund_ticker": "BBB", "constituent_ticker": "NVDA", "weight_pct": 2.0, "as_of": "2026-09-30", "source_name": "Issuer B", "source_url": "https://example.com/b"},
        ])

    def test_overlap_sums_smaller_shared_weights_without_normalizing(self):
        summary, details = calculate_etf_overlap(self.holdings)
        self.assertEqual(summary.iloc[0]["ETF A"], "AAA")
        self.assertEqual(summary.iloc[0]["ETF B"], "BBB")
        self.assertAlmostEqual(summary.iloc[0]["Overlap %"], 7.0)
        self.assertEqual(summary.iloc[0]["Shared constituents"], 2)
        self.assertAlmostEqual(details["Overlap contribution %"].sum(), 7.0)

    def test_coverage_flags_partial_dataset(self):
        out = holdings_weight_coverage(self.holdings)
        self.assertEqual(len(out), 2)
        self.assertTrue((out["coverage_status"].str.contains("Partial/unknown coverage")).all())


if __name__ == "__main__":
    unittest.main()
