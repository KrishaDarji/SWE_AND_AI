import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import health_closedissues as h


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.1", "ACC+": "0.9", "PRED40+": "0.8", "N_estimators": "10", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"},
            {"MRE-": "0.8", "ACC+": "0.2", "PRED40+": "0.1", "N_estimators": "20", "criterion": "b", "Min_sample_leaves": "2", "Min_impurity_decrease": "1", "Max_depth": "4"},
            {"MRE-": "0.2", "ACC+": "0.8", "PRED40+": "0.7", "N_estimators": "15", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0.5", "Max_depth": "3"},
        ]

    def test_exact_prediction(self):
        self.assertEqual(h.predict(self.rows, self.rows[0]), {"MRE-": 0.1, "ACC+": 0.9, "PRED40+": 0.8})

    def test_seeded_baseline_is_reproducible(self):
        a = h.random_search(self.rows, self.rows, seed=3, budget=2)
        b = h.random_search(self.rows, self.rows, seed=3, budget=2)
        self.assertEqual(a, b)

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("MRE-,ACC+\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                h.read_rows(path)


if __name__ == "__main__":
    unittest.main()
