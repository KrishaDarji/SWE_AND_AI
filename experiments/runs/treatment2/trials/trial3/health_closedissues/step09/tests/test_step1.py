import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import summarize_csv


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "criterion", "depth"])
        writer.writerow(["1", "0.5", "a", "2"])
        writer.writerow(["3", "1.0", "b", "4"])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_quality_and_configuration_summary(self):
        result = summarize_csv(self.file.name)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["quality_metrics"]["MRE-"]["mean"], 2.0)
        self.assertEqual(result["configuration"]["criterion"]["type"], "categorical")
        self.assertEqual(result["configuration"]["depth"]["max"], 4.0)

    def test_explicit_quality_column(self):
        result = summarize_csv(self.file.name, quality_columns=["depth"])
        self.assertIn("depth", result["quality_metrics"])
        self.assertIn("MRE-", result["configuration"])

    def test_cli_emits_json(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"), self.file.name]
        output = subprocess.check_output(command, text=True)
        self.assertEqual(json.loads(output)["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
