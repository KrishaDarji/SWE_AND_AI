import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))

from project_summary import load_projects, spot_inconsistent


class SpotTests(unittest.TestCase):
    def test_spots_highest_historical_deviation(self):
        result = spot_inconsistent(load_projects(str(DATA)))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].project_id, "89")
        self.assertGreater(result[0].robust_score, 0)

    def test_top_is_sorted_and_validated(self):
        result = spot_inconsistent(load_projects(str(DATA)), top=4)
        self.assertEqual(len(result), 4)
        self.assertEqual([x.robust_score for x in result],
                         sorted((x.robust_score for x in result), reverse=True))
        with self.assertRaises(ValueError):
            spot_inconsistent([], top=1)
        with self.assertRaises(ValueError):
            spot_inconsistent(load_projects(str(DATA)), top=0)

    def test_cli_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "spot.py"), str(DATA)], text=True)
        parsed = json.loads(output)
        self.assertEqual(parsed[0]["project_id"], "89")


if __name__ == "__main__":
    unittest.main()
