import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from fix_skill import load_dataset, evaluate

class FixSkillTests(unittest.TestCase):
    def test_real_dataset_validates_and_has_recommendation(self):
        p = Path(__file__).parents[8] / "moot/optimize/hpo/Health-ClosedIssues0000.csv"
        rows, rejected = load_dataset(p)
        self.assertEqual(len(rows), 10000); self.assertEqual(rejected, [])
        baseline, best, count, _, _ = evaluate(rows)
        self.assertGreater(count, 0); self.assertIsNotNone(best)
        self.assertGreater(best[0], 0)
        self.assertEqual(best[3], "criterion")

    def test_invalid_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            fields = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                w.writerow(dict(zip(fields, ["0.1","0.2","0.3","10","squared_error","1","0","5"])))
                w.writerow(dict(zip(fields, ["nan","0.2","0.3","10","squared_error","1","0","5"])))
            rows, rejected = load_dataset(p); self.assertEqual(len(rows), 1); self.assertEqual(len(rejected), 1)

if __name__ == "__main__": unittest.main()
