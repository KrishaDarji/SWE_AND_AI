import csv
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarize import load_projects, render, summarize


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.DictWriter(self.file, fieldnames=["YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"])
        writer.writeheader()
        writer.writerows([
            {"YearX": 2000, "Kloc+": 10, "Effort-": 20, "Defects-": 4, "MonthsX": 10},
            {"YearX": 2000, "Kloc+": 20, "Effort-": 40, "Defects-": 8, "MonthsX": 20},
            {"YearX": 2001, "Kloc+": 30, "Effort-": 60, "Defects-": 12, "MonthsX": 30},
        ])
        self.file.close()

    def tearDown(self):
        Path(self.file.name).unlink()

    def test_load_and_aggregate(self):
        summary = summarize(load_projects(self.file.name))
        self.assertEqual(summary["overall"]["count"], 3)
        self.assertEqual(summary["year_counts"], {2000: 2, 2001: 1})
        self.assertAlmostEqual(summary["overall"]["months_mean"], 20)
        self.assertEqual(summary["duration_counts"], {"long (>24 months)": 1,
                                                         "medium (12-24 months)": 1,
                                                         "short (<=12 months)": 1})

    def test_render_contains_outcome_sections(self):
        text = render(summarize(load_projects(self.file.name)))
        self.assertIn("Overall outcomes", text)
        self.assertIn("Delivery-duration bands", text)


if __name__ == "__main__":
    unittest.main()
