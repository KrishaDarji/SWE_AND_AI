import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import compare_with_historical_baseline


class HistoricalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["MRE-", "ACC+", "criterion", "depth"])
        writer.writerows([[5, .5, "old", 2], [1, .9, "new", 4], [2, .8, "mid", 3]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_selector_beats_first_historical_row(self):
        result = compare_with_historical_baseline(self.file.name, neighbors=1)
        self.assertEqual(result["baseline"]["row_index"], 0)
        self.assertEqual(result["selected"]["configuration"]["criterion"], "new")
        self.assertTrue(result["all_metrics_improved"])
        self.assertEqual(result["deltas"]["MRE-"]["delta"], -4.0)

    def test_cli_and_existing_selector_path(self):
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   self.file.name, "--compare-baseline", "--neighbors", "1"]
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result["metrics_improved"], 2)


if __name__ == "__main__":
    unittest.main()
