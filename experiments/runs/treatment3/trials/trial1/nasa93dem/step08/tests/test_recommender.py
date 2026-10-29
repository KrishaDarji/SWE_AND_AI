import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from recommender import read_rows, recommend


DATASET = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv"


class RecommenderTests(unittest.TestCase):
    def test_dataset_is_read_and_has_expected_size(self):
        fields, rows = read_rows(DATASET)
        self.assertIn("Defects-", fields)
        self.assertEqual(len(rows), 93)

    def test_recommendation_is_structured_and_cheap(self):
        result = recommend(DATASET)
        self.assertIn(result["status"], {"change_recommended", "no_improving_change_found"})
        rec = result["recommendation"]
        self.assertLessEqual(rec["cost"], 1)
        if result["status"] == "change_recommended":
            self.assertGreater(rec["risk_reduction"], 0)
            self.assertIsNotNone(rec["feature"])

    def test_missing_target_fails(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=["prec"])
            writer.writeheader(); writer.writerow({"prec": "n"})
            name = handle.name
        try:
            with self.assertRaises(ValueError): recommend(name)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
