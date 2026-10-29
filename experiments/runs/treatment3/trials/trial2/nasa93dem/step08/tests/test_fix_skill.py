import csv, pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from fix_skill import load_data, run

DATA = pathlib.Path(__file__).resolve().parents[8] / "moot/optimize/process/nasa93dem.csv"

class FixSkillTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = load_data(DATA); self.assertEqual(len(rows), 93); self.assertIn("Defects-", rows[0])
    def test_candidates_are_one_step_and_recommendation_is_cheap(self):
        result = run(DATA); self.assertGreater(result["candidate_count"], 0)
        self.assertIsNotNone(result["chosen"]); self.assertEqual(result["chosen"]["cost"], 1)
        self.assertGreater(result["chosen"]["reduction"], 0)
    def test_uncertainty_is_measured(self):
        self.assertGreater(run(DATA)["cv_mae"], 0)

if __name__ == "__main__": unittest.main()
