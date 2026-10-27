import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import blame_hyperparameters


class BlameTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.1", "depth": "2", "criterion": "good"},
            {"MRE-": "0.2", "depth": "3", "criterion": "good"},
            {"MRE-": "0.8", "depth": "8", "criterion": "bad"},
            {"MRE-": "0.9", "depth": "9", "criterion": "bad"},
        ]

    def test_numeric_and_categorical_associations_are_ranked(self):
        result = blame_hyperparameters(self.rows, configuration_columns=["depth", "criterion"])
        self.assertEqual(result["high_error_count"], 2)
        self.assertEqual(result["high_error_threshold"], 0.8)
        self.assertEqual(result["associations"][0]["hyperparameter"], "criterion")
        numeric = next(item for item in result["associations"] if item["hyperparameter"] == "depth")
        self.assertGreater(numeric["pearson_high_error"], 0.0)

    def test_invalid_quantile_and_missing_error_are_rejected(self):
        with self.assertRaises(ValueError):
            blame_hyperparameters(self.rows, high_error_quantile=1.1)
        with self.assertRaises(ValueError):
            blame_hyperparameters(self.rows, error_column="missing")


if __name__ == "__main__":
    unittest.main()
