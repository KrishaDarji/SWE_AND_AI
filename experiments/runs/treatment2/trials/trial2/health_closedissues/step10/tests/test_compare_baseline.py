import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import compare_with_historical_baseline


class HistoricalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.80", "ACC+": "0.40", "PRED40+": "0.30", "depth": "3"},
            {"MRE-": "0.20", "ACC+": "0.90", "PRED40+": "0.80", "depth": "5"},
            {"MRE-": "0.50", "ACC+": "0.60", "PRED40+": "0.50", "depth": "7"},
        ]

    def test_uses_same_objective_and_reports_selector_gain(self):
        result = compare_with_historical_baseline(self.rows, {"depth": "3"}, ["depth"])
        self.assertEqual(result["selector"]["configuration"], {"depth": "5"})
        self.assertTrue(result["selector_better"])
        self.assertGreater(result["score_difference_selector_minus_historical"], 0)
        self.assertEqual(result["objective"], "weighted normalized MRE-/ACC+/PRED40+")

    def test_requires_observed_historical_configuration(self):
        with self.assertRaises(ValueError):
            compare_with_historical_baseline(self.rows, {"depth": "99"}, ["depth"])


if __name__ == "__main__":
    unittest.main()
