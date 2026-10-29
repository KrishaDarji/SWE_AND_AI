import csv
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from blame import analyze, prepare, read_dataset


DATASET = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv"


class BlameTests(unittest.TestCase):
    def test_dataset_has_expected_target_and_rows(self):
        rows = read_dataset(DATASET)
        self.assertGreater(len(rows), 50)
        self.assertIn("Effort-", rows[0])

    def test_prepare_marks_above_median(self):
        rows = read_dataset(DATASET)
        _, y, names, categorical, numeric, cutoff, values = prepare(rows)
        self.assertEqual(len(y), len(rows))
        self.assertEqual(sum(y), sum(value > cutoff for value in values))
        self.assertIn("Kloc+", names)
        self.assertIn("Kloc+", numeric)
        self.assertIn("rely", categorical)

    def test_analysis_is_ranked_and_finite(self):
        result = analyze(DATASET)
        self.assertEqual(result["rows"], 93)
        self.assertEqual(len(result["factors"]), 25)
        self.assertEqual(result["factors"], sorted(result["factors"], key=lambda x: x["importance"], reverse=True))
        self.assertTrue(0 <= result["cross_validated_accuracy"] <= 1)
        for item in result["factors"]:
            self.assertTrue(abs(item["importance"]) < 10)


if __name__ == "__main__":
    unittest.main()
