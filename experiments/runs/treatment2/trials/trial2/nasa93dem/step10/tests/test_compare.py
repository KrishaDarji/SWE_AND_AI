import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import compare_estimators, load_projects


class ComparisonTests(unittest.TestCase):
    def test_same_protocol_compares_both_methods(self):
        result = compare_estimators(load_projects(str(DATA)), neighbors=5)
        self.assertEqual(result.projects_evaluated, 93)
        self.assertEqual(result.metric, "MAPE (%)")
        self.assertGreater(result.estimator_mape, 0)
        self.assertGreater(result.baseline_mape, 0)
        self.assertEqual(result.estimator_better, result.estimator_mape < result.baseline_mape)

    def test_validation(self):
        with self.assertRaises(ValueError):
            compare_estimators(load_projects(str(DATA))[:1])
        with self.assertRaises(ValueError):
            compare_estimators(load_projects(str(DATA)), neighbors=0)

    def test_cli_emits_comparison_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "compare.py"), str(DATA)], text=True)
        parsed = json.loads(output)
        self.assertIn("baseline_mape", parsed)


if __name__ == "__main__":
    unittest.main()
