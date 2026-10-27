import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import blame_factors, load_projects


class BlameTests(unittest.TestCase):
    def test_ranks_factor_values_and_uses_median_default(self):
        result = blame_factors(load_projects(str(DATA)), top=5)
        self.assertEqual(result.project_count, 93)
        self.assertEqual(len(result.factors), 5)
        self.assertGreaterEqual(result.excessive_projects, 1)
        self.assertTrue(all(result.factors[i].score >= result.factors[i + 1].score
                             for i in range(len(result.factors) - 1)))

    def test_explicit_threshold_and_validation(self):
        result = blame_factors(load_projects(str(DATA)), threshold=1000, top=2,
                               minimum_group_size=5)
        self.assertEqual(result.effort_threshold, 1000.0)
        with self.assertRaises(ValueError):
            blame_factors([], top=1)

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "blame.py"), str(DATA), "--top", "3"],
            text=True)
        self.assertEqual(len(json.loads(output)["factors"]), 3)


if __name__ == "__main__":
    unittest.main()
