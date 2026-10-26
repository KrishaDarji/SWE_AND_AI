import csv
import os
import tempfile
import unittest

from source.justify_config import read_rows, recommend


FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
          "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


class RecommendationTests(unittest.TestCase):
    def test_balanced_row_wins(self):
        rows = [
            {"MRE-": 0.0, "ACC+": 0.0, "PRED40+": 0.0, "N_estimators": 10,
             "criterion": "a", "Min_sample_leaves": 1, "Min_impurity_decrease": 0.0, "Max_depth": 3},
            {"MRE-": 0.5, "ACC+": 0.5, "PRED40+": 0.5, "N_estimators": 20,
             "criterion": "b", "Min_sample_leaves": 2, "Min_impurity_decrease": 0.0, "Max_depth": 5},
            {"MRE-": 1.0, "ACC+": 1.0, "PRED40": 1.0, "N_estimators": 30,
             "criterion": "c", "Min_sample_leaves": 3, "Min_impurity_decrease": 0.0, "Max_depth": 8},
        ]
        rows[2]["PRED40+"] = rows[2].pop("PRED40")
        result = recommend(rows)
        self.assertEqual(result["source_row"], 4)

    def test_read_rows_and_empty_rejection(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow({"MRE-": "0.1", "ACC+": "0.8", "PRED40+": "0.7",
                             "N_estimators": "10", "criterion": "x", "Min_sample_leaves": "2",
                             "Min_impurity_decrease": "0", "Max_depth": "4"})
            path = handle.name
        try:
            self.assertEqual(len(read_rows(path)), 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
