import csv
import os
import tempfile
import unittest

from source.spot import read_records, spot


class SpotTests(unittest.TestCase):
    def test_reads_dataset(self):
        path = os.environ["NASA_DATASET"]
        fields, rows = read_records(path)
        self.assertIn("Kloc+", fields)
        self.assertGreater(len(rows), 90)

    def test_returns_a_ranked_project(self):
        result = spot(os.environ["NASA_DATASET"])
        self.assertIn("idX", result["record"])
        self.assertEqual(result["row_number"], int(result["row_number"]))
        self.assertEqual(len(result["ranked_scores"]), 93)
        self.assertLessEqual(result["anomaly_score"], result["ranked_scores"][-1]["score"])


if __name__ == "__main__":
    unittest.main()
