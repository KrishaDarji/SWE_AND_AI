import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import spot_quality_anomalies


class AnomalySpottingTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "depth", "criterion"])
        writer.writerows([[1, .9, 2, "a"], [1.1, .91, 2, "a"],
                          [1.2, .89, 3, "b"], [100, .2, 3, "b"],
                          ["bad", .8, 4, "c"]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_reports_nonfinite_outlier_and_duplicate_disagreement(self):
        result = spot_quality_anomalies(self.file.name)
        reasons = {(item["quality_column"], item["reason"]) for item in result["anomalies"]}
        self.assertIn(("MRE-", "missing_or_nonfinite"), reasons)
        self.assertIn(("MRE-", "robust_outlier"), reasons)
        self.assertEqual(len(result["duplicate_configuration_conflicts"]), 4)

    def test_rejects_unknown_column_and_cli(self):
        with self.assertRaises(ValueError):
            spot_quality_anomalies(self.file.name, ["missing"])
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--spot-anomalies"]
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["row_count"], 5)


if __name__ == "__main__":
    unittest.main()
