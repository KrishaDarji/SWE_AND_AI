import csv
import json
import os
import tempfile
import unittest

from source.blame import TARGET, analyze, group_summary, read_data


class BlameTests(unittest.TestCase):
    def setUp(self):
        self.handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        writer = csv.writer(self.handle)
        writer.writerow([TARGET, "alpha", "criterion"])
        writer.writerows([[1, 1, "a"], [2, 2, "a"], [3, 3, "b"], [4, 4, "b"], [5, 5, "b"], [6, 6, "b"]])
        self.handle.close()

    def tearDown(self):
        os.unlink(self.handle.name)

    def test_read_data_and_group_summary(self):
        rows, features = read_data(self.handle.name)
        self.assertEqual(features, ["alpha", "criterion"])
        summary = group_summary(rows, "criterion", 4.75)
        self.assertEqual(summary[0]["value"], "b")
        self.assertGreater(summary[0]["mean_error"], summary[1]["mean_error"])

    def test_analyze_returns_rankings_and_metrics(self):
        result = analyze(self.handle.name)
        self.assertEqual(result["rows"], 6)
        self.assertEqual(len(result["permutation_importance"]), 2)
        self.assertIn("test_mae", result["model"])


if __name__ == "__main__":
    unittest.main()
