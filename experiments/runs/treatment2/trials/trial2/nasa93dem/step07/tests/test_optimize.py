import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import choose_settings, load_projects


class OptimizationTests(unittest.TestCase):
    def test_selects_feasible_settings_and_ranks_efficiency(self):
        result = choose_settings(load_projects(str(DATA)), 1000, ["sced"], top=5)
        self.assertTrue(result.selected.feasible)
        self.assertLessEqual(result.selected.mean_effort, 1000)
        self.assertEqual(result.settings_columns, ["sced"])
        self.assertGreater(result.selected.value_per_effort, 0)

    def test_no_feasible_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_settings(load_projects(str(DATA)), .001, ["sced"])

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "optimize.py"), str(DATA), "1000",
             "--settings", "sced", "--top", "3"], text=True)
        parsed = json.loads(output)
        self.assertIn("selected", parsed)
        self.assertLessEqual(parsed["selected"]["mean_effort"], 1000)


if __name__ == "__main__":
    unittest.main()
