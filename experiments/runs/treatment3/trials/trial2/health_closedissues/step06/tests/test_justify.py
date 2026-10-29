import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify import load_dataset, recommend, utility

DATA = Path(__file__).parents[7] / "moot/optimize/hpo/Health-ClosedIssues0000.csv"

class JustifyTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = load_dataset(DATA)
        self.assertEqual(len(rows), 10000)
        self.assertTrue(all(0 <= utility(r) <= 1 for r in rows))

    def test_recommendation_is_observed_and_reproducible(self):
        rows = load_dataset(DATA)
        a = recommend(rows, bootstraps=100)
        b = recommend(rows, bootstraps=100)
        self.assertEqual(a[0], b[0]); self.assertIn(a[0], rows)

    def test_rejects_bad_outcome(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as f:
            fields = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(dict.fromkeys(fields, "0"))
            name = f.name
        with self.assertRaises(ValueError): load_dataset(name)

if __name__ == "__main__": unittest.main()
