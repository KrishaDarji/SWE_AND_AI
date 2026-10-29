import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from justify import load, run


DATASET = ROOT.parents[6] / "moot" / "optimize" / "hpo" / "Health-ClosedIssues0000.csv"


class JustifyTests(unittest.TestCase):
    def test_dataset_is_loaded_with_expected_contract(self):
        rows, x, y = load(DATASET)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(x.shape, (10000, 5))
        self.assertEqual(y.shape, (10000, 3))

    def test_run_is_deterministic_and_has_required_evidence(self):
        a = run(DATASET)
        b = run(DATASET)
        self.assertEqual(a["selected_configuration"], b["selected_configuration"])
        self.assertEqual(a["split"], {"train": 8000, "test": 2000, "seed": 31415})
        self.assertLess(a["evaluations"]["random_forest"]["utility_mae"], 0.20)
        self.assertGreater(a["evaluations"]["random_forest"]["utility_r2"], 0.0)
        self.assertIn("extra_trees", a["evaluations"])


if __name__ == "__main__":
    unittest.main()
