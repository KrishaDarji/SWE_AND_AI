import csv, math, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify import justify

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class JustifyTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False); self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name
    def row(self, mre, acc, pred, i):
        return {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": str(i), "criterion": "x", "Min_sample_leaves": str(i), "Min_impurity_decrease": str(i), "Max_depth": str(i)}
    def test_policy_ties_and_reference_arithmetic(self):
        got = justify(self.write([self.row(1, 2, 3, 1), self.row(1, 4, 2, 2), self.row(1, 4, 5, 3), self.row(3, 0, 0, 4)]))
        self.assertEqual(got["selected"]["record_id"], 3)
        self.assertEqual(got["selected"]["configuration"]["Max_depth"], "3")
        self.assertEqual(got["dataset_medians"], {"MRE-": 1.0, "ACC+": 3.0, "PRED40+": 2.5})
        self.assertEqual(got["selected_minus_median"], {"MRE-": 0.0, "ACC+": 1.0, "PRED40+": 2.5})
    def test_missing_nonfinite_and_earliest_tie(self):
        got = justify(self.write([self.row("nan", 9, 9, 1), self.row(2, "bad", 4, 2), self.row(2, "", 4, 3)]))
        self.assertEqual(got["selected"]["record_id"], 2)
        self.assertEqual(got["missing_nonfinite"], {"MRE-": 1, "ACC+": 2, "PRED40+": 0})
        self.assertEqual(got["median_valid_counts"], {"MRE-": 2, "ACC+": 0, "PRED40+": 2})
    def test_no_finite_mre(self):
        with self.assertRaises(ValueError): justify(self.write([self.row("nan", 1, 1, 1)]))

if __name__ == "__main__": unittest.main()
