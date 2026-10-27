import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import justify_configuration


class JustificationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.10", "ACC+": "0.90", "depth": "3", "criterion": "good"},
            {"MRE-": "0.20", "ACC+": "0.85", "depth": "4", "criterion": "good"},
            {"MRE-": "0.80", "ACC+": "0.40", "depth": "8", "criterion": "bad"},
            {"MRE-": "0.90", "ACC+": "0.35", "depth": "9", "criterion": "bad"},
        ]

    def test_recommends_candidate_better_than_baseline(self):
        result = justify_configuration(self.rows, {"depth": "3", "criterion": "good"},
                                       configuration_columns=["depth", "criterion"])
        self.assertEqual(result["recommendation"], "recommend")
        self.assertGreater(result["deltas"]["error_reduction"], 0)
        self.assertGreater(result["deltas"]["accuracy_gain"], 0)
        self.assertTrue(result["caveat"])

    def test_rejects_missing_candidate_parameter(self):
        with self.assertRaises(ValueError):
            justify_configuration(self.rows, {"depth": "3"},
                                   configuration_columns=["depth", "criterion"])


if __name__ == "__main__":
    unittest.main()
