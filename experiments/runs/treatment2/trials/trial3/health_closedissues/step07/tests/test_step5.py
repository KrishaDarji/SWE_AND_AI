import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import blame_high_error


class BlameTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "depth", "criterion"])
        writer.writerows([[1, 1, "a"], [2, 2, "a"], [8, 8, "b"], [9, 9, "b"]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_numeric_and_categorical_associations(self):
        result = blame_high_error(self.file.name)
        self.assertEqual(result["high_error_count"], 1)
        self.assertIn(result["associations"][0]["hyperparameter"], {"depth", "criterion"})
        criterion = next(x for x in result["associations"] if x["hyperparameter"] == "criterion")
        self.assertEqual(criterion["levels"][1]["value"], "b")
        self.assertEqual(criterion["levels"][1]["high_error_rate"], 0.5)

    def test_cli_blame(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--blame"]
        self.assertEqual(json.loads(subprocess.check_output(command, text=True))["row_count"], 4)


if __name__ == "__main__":
    unittest.main()
