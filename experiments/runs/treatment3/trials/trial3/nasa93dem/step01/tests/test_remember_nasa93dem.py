import csv
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import remember_nasa93dem as analysis


DATASET = pathlib.Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"


class AnalysisTests(unittest.TestCase):
    def test_load_schema_and_row_count(self):
        headers, rows = analysis.load_csv(str(DATASET.resolve()))
        self.assertEqual(len(rows), 93)
        self.assertTrue(set(analysis.OUTCOMES).issubset(headers))

    def test_outcomes_are_numeric(self):
        headers, rows = analysis.load_csv(str(DATASET.resolve()))
        for outcome in analysis.OUTCOMES:
            values, bad = analysis.numeric(rows, outcome)
            self.assertEqual(len(values), 93)
            self.assertEqual(bad, 0)

    def test_rating_order_is_monotonic(self):
        self.assertLess(analysis.RATING_ORDER["vl"], analysis.RATING_ORDER["l"])
        self.assertLess(analysis.RATING_ORDER["l"], analysis.RATING_ORDER["h"])


if __name__ == "__main__":
    unittest.main()
