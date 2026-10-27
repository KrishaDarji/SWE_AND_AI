import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import summarize_csv, predict_quality, certify_families

class Step3Tests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "family", "depth"])
        writer.writerow(["1", ".9", "a", "2"])
        writer.writerow(["1.2", ".8", "a", "3"])
        writer.writerow(["3", ".5", "b", "2"])
        writer.writerow(["3.2", ".4", "b", "3"])
        self.file.close()
    def tearDown(self): os.unlink(self.file.name)
    def test_step1_summary(self):
        self.assertEqual(summarize_csv(self.file.name)["row_count"], 4)
    def test_step2_prediction(self):
        result = predict_quality(self.file.name, {"family": "a", "depth": 2}, neighbors=1)
        self.assertEqual(result["predicted_quality"]["MRE-"], 1.0)
    def test_certification_reports_pair(self):
        result = certify_families(self.file.name, ["family"])
        self.assertEqual(result["comparison_count"], 1)
        pair = result["comparisons"][0]
        self.assertEqual(pair["reliably_better"], "a")
        self.assertTrue(pair["metrics"]["MRE-"]["reliable"])
    def test_cli_certification(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--family-column", "family"]
        self.assertEqual(json.loads(subprocess.check_output(command, text=True))["comparison_count"], 1)

if __name__ == "__main__": unittest.main()
