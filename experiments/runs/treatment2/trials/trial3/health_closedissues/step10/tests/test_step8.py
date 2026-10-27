import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import suggest_smallest_change


class SmallestChangeTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "depth", "criterion"])
        writer.writerows([[1.0, .9, 2, "a"], [1.5, .8, 3, "a"],
                          [.8, .7, 2, "b"], [2.0, .6, 4, "b"]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_recommends_single_change_from_best_baseline(self):
        result = suggest_smallest_change(self.file.name)
        self.assertEqual(result["baseline_error"], 1.0)
        self.assertEqual(result["recommendation"]["hyperparameter"], "criterion")

    def test_finds_improving_one_change(self):
        with open(self.file.name, "a", newline="") as handle:
            csv.writer(handle).writerow([.5, .95, 3, "a"])
        result = suggest_smallest_change(self.file.name)
        self.assertEqual(result["recommendation"]["hyperparameter"], "depth")
        self.assertEqual(result["recommendation"]["to"], "3")
        self.assertEqual(result["recommendation"]["improvement"], .5)

    def test_cli(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--suggest-smallest-change"]
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["selection"], "smallest observed one-hyperparameter error improvement")


if __name__ == "__main__":
    unittest.main()
