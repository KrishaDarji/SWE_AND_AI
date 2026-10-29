import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("choose", ROOT / "source" / "choose.py")
choose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choose)


class ChooseTests(unittest.TestCase):
    def test_loads_real_dataset_and_selects(self):
        rows, rejected, _ = choose.load_rows(Path("moot/optimize/hpo/Health-ClosedIssues0000.csv"))
        self.assertEqual(rejected, 0)
        best, ranked = choose.choose(rows)
        self.assertEqual(len(ranked), len(rows))
        self.assertIn("composite_score", best)

    def test_rejects_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=choose.REQUIRED)
                writer.writeheader()
                writer.writerow({"MRE-": 2, "ACC+": .5, "PRED40+": .5,
                                 "N_estimators": 10, "criterion": "squared_error",
                                 "Min_sample_leaves": 1, "Min_impurity_decrease": 0,
                                 "Max_depth": 3})
            with self.assertRaises(ValueError):
                choose.load_rows(path)


if __name__ == "__main__":
    unittest.main()
