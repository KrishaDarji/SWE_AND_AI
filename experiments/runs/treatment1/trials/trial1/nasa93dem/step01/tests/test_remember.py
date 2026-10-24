import csv
import os
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.remember import load_projects, render_report, summarize


class RememberTests(unittest.TestCase):
    def setUp(self):
        self.path = str(Path(__file__).resolve().parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv")

    def test_loads_expected_dataset_and_numeric_outcomes(self):
        projects = load_projects(self.path)
        self.assertGreater(len(projects), 90)
        self.assertIsInstance(projects[0]["Effort-"], float)

    def test_summary_has_consistent_statistics(self):
        summary = summarize(load_projects(self.path))
        self.assertEqual(summary["project_count"], summary["outcomes"]["Effort-"]["count"])
        self.assertLessEqual(summary["outcomes"]["Effort-"]["min"], summary["outcomes"]["Effort-"]["median"])
        self.assertIn("prec", summary["characteristics"]["drivers"])

    def test_rejects_missing_required_field(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as stream:
            writer = csv.writer(stream)
            writer.writerow(["idX"])
            writer.writerow(["1"])
            path = stream.name
        try:
            with self.assertRaises(ValueError):
                load_projects(path)
        finally:
            os.unlink(path)

    def test_report_mentions_effort(self):
        report = render_report(summarize(load_projects(self.path)))
        self.assertIn("Effort-", report)
        self.assertIn("Projects:", report)


if __name__ == "__main__":
    unittest.main()
