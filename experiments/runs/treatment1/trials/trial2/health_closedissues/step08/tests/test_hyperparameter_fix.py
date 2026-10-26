import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source.hyperparameter_fix import suggest, read_rows


FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
          "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def row(error, n="40", criterion="squared_error", leaves="3", impurity="2.75", depth="10"):
    return {"MRE-": str(error), "ACC+": "0", "PRED40+": "0",
            "N_estimators": n, "criterion": criterion,
            "Min_sample_leaves": leaves, "Min_impurity_decrease": impurity,
            "Max_depth": depth}


class SuggestionTests(unittest.TestCase):
    def test_picks_smallest_improving_numeric_change(self):
        result = suggest([row(.50), row(.40, n="41"), row(.30, n="42"),
                          row(.20, depth="8")])
        self.assertEqual(result["parameter"], "N_estimators")
        self.assertEqual(result["after"], "41")

    def test_categorical_change_is_considered(self):
        result = suggest([row(.50), row(.40, criterion="absolute_error")])
        self.assertEqual(result["parameter"], "criterion")

    def test_requires_exactly_one_change_and_improvement(self):
        self.assertIsNone(suggest([row(.50), row(.40, n="41", depth="9"), row(.60, n="41")]))

    def test_reads_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(row(.5))
            self.assertEqual(len(read_rows(path)), 1)


if __name__ == "__main__":
    unittest.main()
