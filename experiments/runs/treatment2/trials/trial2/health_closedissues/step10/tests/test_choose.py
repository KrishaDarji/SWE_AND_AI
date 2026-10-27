import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import choose_configuration


class ChooseConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.10", "ACC+": "0.80", "PRED40+": "0.70", "depth": "3"},
            {"MRE-": "0.20", "ACC+": "0.95", "PRED40+": "0.90", "depth": "5"},
            {"MRE-": "0.05", "ACC+": "0.60", "PRED40+": "0.50", "depth": "2"},
        ]

    def test_balanced_score_selects_best_candidate(self):
        result = choose_configuration(self.rows, ["depth"])
        self.assertEqual(result["chosen"]["configuration"], {"depth": "5"})
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(len(result["ranked"]), 3)

    def test_error_metric_is_lower_is_better(self):
        result = choose_configuration(self.rows, ["depth"], weights=(1, 0, 0))
        self.assertEqual(result["chosen"]["configuration"], {"depth": "2"})

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_configuration(self.rows, ["depth"], weights=(1, 1))
        with self.assertRaises(ValueError):
            choose_configuration(self.rows, ["depth"], prediction_column="missing")


if __name__ == "__main__":
    unittest.main()
