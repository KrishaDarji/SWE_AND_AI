import csv
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import health_closedissues as hc


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.5", "ACC+": "0.5", "PRED40+": "0.5", "N_estimators": "10", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "5"},
            {"MRE-": "0.2", "ACC+": "0.8", "PRED40+": "0.7", "N_estimators": "20", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "5"},
        ]
        self.baseline = {"N_estimators": 10, "criterion": "a", "Min_sample_leaves": 1, "Min_impurity_decrease": 0, "Max_depth": 5}

    def test_exact_prediction(self):
        self.assertEqual(hc.predict(self.rows, self.baseline), {"MRE-": .5, "ACC+": .5, "PRED40+": .5})

    def test_suggests_improving_nearest_row(self):
        candidate, prediction, improved = hc.suggest(self.rows, self.baseline)
        self.assertTrue(improved)
        self.assertEqual(candidate["N_estimators"], "20")
        self.assertLess(prediction["MRE-"], .5)

    def test_read_rows_rejects_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            name = handle.name
        try:
            with self.assertRaises(ValueError):
                hc.read_rows(name)
        finally:
            Path(name).unlink()


if __name__ == "__main__":
    unittest.main()
