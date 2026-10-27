import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import load_projects, recommend_process_change


class RiskTests(unittest.TestCase):
    def test_recommends_a_real_risk_reducing_change(self):
        result = recommend_process_change(load_projects(str(DATA)), top=5)
        change = result.recommendation
        self.assertGreater(change.risk_reduction, 0)
        self.assertLess(change.estimated_risk, change.baseline_risk)
        self.assertEqual(len(result.candidates), 5)

    def test_candidate_order_is_cheapest_first(self):
        result = recommend_process_change(load_projects(str(DATA)), top=5)
        costs = [c.change_cost for c in result.candidates]
        self.assertEqual(costs, sorted(costs))

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "risk.py"), str(DATA), "--top", "3"],
            text=True)
        parsed = json.loads(output)
        self.assertIn("recommendation", parsed)


if __name__ == "__main__":
    unittest.main()
