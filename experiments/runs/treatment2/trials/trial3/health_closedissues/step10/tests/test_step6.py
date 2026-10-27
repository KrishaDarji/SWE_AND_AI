import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import justify_configuration


class JustificationTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "criterion", "depth"])
        writer.writerows([[1, .9, "a", 2], [2, .8, "a", 3], [8, .2, "b", 2], [9, .1, "b", 3]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_recommends_observed_configuration_and_evidence(self):
        result = justify_configuration(self.file.name)
        self.assertEqual(result["selection"], "best observed normalized quality rank")
        self.assertEqual(result["recommended_configuration"]["criterion"], "a")
        self.assertIn("predicted_quality", result)
        self.assertIn("associations", result["high_error_analysis"])

    def test_cli_accepts_explicit_configuration(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--justify", "--predict-config", '{"criterion":"a","depth":2}']
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["selection"], "provided")
        self.assertEqual(result["recommended_configuration"]["depth"], 2)


if __name__ == "__main__":
    unittest.main()
