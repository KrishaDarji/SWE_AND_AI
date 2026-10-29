import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = next(p for p in ROOT.parents if (p / "moot").is_dir())
sys.path.insert(0, str(ROOT / "source"))
from race_skill import load_dataset, evaluate, utility


class RaceSkillTests(unittest.TestCase):
    def test_real_dataset_validates(self):
        rows = load_dataset(REPO / "moot/optimize/hpo/Health-ClosedIssues0000.csv")
        self.assertEqual(len(rows), 10000)
        self.assertTrue(all(set(("MRE-", "ACC+", "PRED40+")) <= set(r) for r in rows))

    def test_utility_prefers_better_metrics(self):
        values = [{"MRE-": 0, "ACC+": 1, "PRED40+": 1}, {"MRE-": 1, "ACC+": 0, "PRED40+": 0}]
        self.assertGreater(utility(values)[0], utility(values)[1])

    def test_evaluation_is_reproducible_shape(self):
        rows = load_dataset(REPO / "moot/optimize/hpo/Health-ClosedIssues0000.csv")
        results, _ = evaluate(rows, 6000, 6100)
        self.assertEqual([r.method for r in results], ["treatment3_selector", "historical_baseline"])
        self.assertEqual([r.n for r in results], [1, 1])


if __name__ == "__main__":
    unittest.main()
