import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

SOURCE = os.path.join(os.path.dirname(__file__), "..", "source", "model_quality_summary.py")
sys.path.insert(0, os.path.dirname(SOURCE))
from model_quality_summary import render_text, summarize


class SummaryTests(unittest.TestCase):
    def make_csv(self):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(handle)
        writer.writerow(["MRE-", "ACC+", "PRED40+", "criterion", "depth"])
        writer.writerow(["1", "0.5", "0", "a", "3"])
        writer.writerow(["3", "0.75", "1", "a", "5"])
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_summary_groups_metrics_and_configurations(self):
        result = summarize(self.make_csv())
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["quality_metrics"]["MRE-"]["mean"], 2.0)
        self.assertEqual(result["configuration_settings"]["criterion"]["values"], {"a": 2})

    def test_text_rendering(self):
        output = render_text(summarize(self.make_csv()))
        self.assertIn("Model quality metrics", output)
        self.assertIn("criterion", output)

    def test_cli_json(self):
        completed = subprocess.run(
            [sys.executable, SOURCE, self.make_csv(), "--json"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(completed.stdout)["rows"], 2)


if __name__ == "__main__":
    unittest.main()
