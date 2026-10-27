import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import justify_estimate, load_projects


class JustificationTests(unittest.TestCase):
    def test_justification_contains_range_evidence_and_caveats(self):
        result = justify_estimate(load_projects(str(DATA)), "P-6", 10, 10000, 10000,
                                  neighbors=5)
        self.assertEqual(result.decision, "approve")
        self.assertEqual(result.historical_projects, 5)
        self.assertEqual(len(result.evidence), 3)
        self.assertIn("upper screening", result.sponsor_message)
        self.assertGreaterEqual(len(result.caveats), 2)

    def test_tight_limits_are_explained(self):
        result = justify_estimate(load_projects(str(DATA)), "P-7", 10, 1, 1,
                                  neighbors=5)
        self.assertEqual(result.decision, "hold")
        self.assertIn("exceeds", result.sponsor_message)

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "justify.py"), str(DATA),
             "P-8", "10", "10000", "10000"], text=True)
        self.assertIn("sponsor_message", json.loads(output))

    def test_invalid_factor_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            justify_estimate(load_projects(str(DATA)), "P-9", 10, 1, 1, top_factors=0)


if __name__ == "__main__":
    unittest.main()
