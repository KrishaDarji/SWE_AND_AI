import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import load_projects, summarize_projects


class SummaryTests(unittest.TestCase):
    def test_loads_all_projects_and_outcomes(self):
        projects = load_projects(str(DATA))
        self.assertEqual(len(projects), 93)
        self.assertEqual(projects[0].project_id, "1")
        self.assertAlmostEqual(projects[0].effort, 117.6)

    def test_summary_has_totals_and_year_cohorts(self):
        summary = summarize_projects(load_projects(str(DATA)), "nasa93dem.csv")
        self.assertEqual(summary.project_count, 93)
        self.assertEqual(summary.year_range, [1971, 1987])
        self.assertEqual(sum(x["project_count"] for x in summary.by_year), 93)
        self.assertGreater(summary.totals["effort"], 0)

    def test_cli_emits_json(self):
        output = subprocess.check_output([sys.executable, str(ROOT / "source" / "summarize.py"), str(DATA)], text=True)
        parsed = json.loads(output)
        self.assertEqual(parsed["project_count"], 93)

    def test_empty_collection_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_projects([], "empty")


if __name__ == "__main__":
    unittest.main()
