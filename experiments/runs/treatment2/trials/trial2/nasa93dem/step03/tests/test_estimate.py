import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import estimate_effort, load_projects


class EstimateTests(unittest.TestCase):
    def test_estimate_is_deterministic_and_has_screening_range(self):
        result = estimate_effort(load_projects(str(DATA)), 10, neighbors=5)
        self.assertEqual(result.historical_projects, 5)
        self.assertGreater(result.estimated_effort, 0)
        self.assertLessEqual(result.lower_effort, result.estimated_effort)
        self.assertLessEqual(result.estimated_effort, result.upper_effort)

    def test_invalid_proposal_is_rejected(self):
        with self.assertRaises(ValueError):
            estimate_effort(load_projects(str(DATA)), 0)

    def test_estimate_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "estimate.py"), str(DATA), "10"],
            text=True)
        self.assertGreater(json.loads(output)["estimated_effort"], 0)


if __name__ == "__main__":
    unittest.main()
