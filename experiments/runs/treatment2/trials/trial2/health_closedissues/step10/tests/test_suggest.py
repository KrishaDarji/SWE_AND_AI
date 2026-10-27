import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import suggest_hyperparameter_change


class SuggestChangeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.80", "ACC+": "0.4", "PRED40+": "0.3", "depth": "3", "criterion": "a"},
            {"MRE-": "0.50", "ACC+": "0.5", "PRED40+": "0.4", "depth": "4", "criterion": "a"},
            {"MRE-": "0.20", "ACC+": "0.9", "PRED40+": "0.8", "depth": "4", "criterion": "b"},
            {"MRE-": "0.10", "ACC+": "1.0", "PRED40+": "0.9", "depth": "9", "criterion": "b"},
        ]

    def test_prefers_smallest_change(self):
        result = suggest_hyperparameter_change(
            self.rows, {"depth": "3", "criterion": "a"})
        self.assertEqual(result["recommendation"], "change")
        self.assertEqual(result["candidate"]["distance"], 1)
        self.assertEqual(result["candidate"]["changed"], {"depth": {"from": "3", "to": "4"}})

    def test_reports_no_improvement_and_validates_baseline(self):
        result = suggest_hyperparameter_change(
            [{"MRE-": "0.1", "depth": "1"}], {"depth": "1"})
        self.assertEqual(result["recommendation"], "no_improvement")
        with self.assertRaises(ValueError):
            suggest_hyperparameter_change(self.rows, {"depth": "missing", "criterion": "a"})


if __name__ == "__main__":
    unittest.main()
