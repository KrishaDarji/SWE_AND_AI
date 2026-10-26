import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from blame import analyze, load_rows, pearson


class BlameTests(unittest.TestCase):
    def test_pearson_direction(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_analysis_identifies_high_error_group(self):
        rows = [
            {"MRE-": "1", "N_estimators": "1", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "1", "criterion": "a"},
            {"MRE-": "2", "N_estimators": "2", "Min_sample_leaves": "2", "Min_impurity_decrease": "0", "Max_depth": "2", "criterion": "b"},
            {"MRE-": "3", "N_estimators": "3", "Min_sample_leaves": "3", "Min_impurity_decrease": "0", "Max_depth": "3", "criterion": "b"},
            {"MRE-": "4", "N_estimators": "4", "Min_sample_leaves": "4", "Min_impurity_decrease": "0", "Max_depth": "4", "criterion": "b"},
            {"MRE-": "5", "N_estimators": "5", "Min_sample_leaves": "5", "Min_impurity_decrease": "0", "Max_depth": "5", "criterion": "b"},
        ]
        result = analyze(rows)
        self.assertEqual(result["high_count"], 2)  # cutoff includes ties at the boundary
        self.assertGreater(result["numeric"]["N_estimators"]["high_mean"], result["numeric"]["N_estimators"]["overall_mean"])
        self.assertGreater(result["categorical"]["b"]["mean_error"], result["categorical"]["a"]["mean_error"])

    def test_load_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["MRE-"])
            writer.writerow(["0.5"])
            name = f.name
        try:
            self.assertEqual(len(load_rows(name)), 1)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
