import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import assess_proposal, load_projects


class AssessmentTests(unittest.TestCase):
    def test_approval_uses_historical_estimate_and_limits(self):
        result = assess_proposal(load_projects(str(DATA)), "P-1", 10, 10000, 10000,
                                 neighbors=5)
        self.assertEqual(result.decision, "approve")
        self.assertEqual(result.reasons, [])

    def test_hold_explains_budget_and_capacity_failures(self):
        result = assess_proposal(load_projects(str(DATA)), "P-2", 10, 1, 1,
                                 neighbors=5)
        self.assertEqual(result.decision, "hold")
        self.assertEqual(len(result.reasons), 2)

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "assess.py"), str(DATA),
             "P-3", "10", "10000", "10000"], text=True)
        self.assertEqual(json.loads(output)["decision"], "approve")

    def test_negative_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            assess_proposal(load_projects(str(DATA)), "P-4", 10, -1, 100)


if __name__ == "__main__":
    unittest.main()
