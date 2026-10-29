import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from blame import rank

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class BlameTests(unittest.TestCase):
    def make(self, rows):
        d = tempfile.TemporaryDirectory(); p = Path(d.name) / "x.csv"
        with p.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        self.addCleanup(d.cleanup); return p
    def row(self, **kw):
        x = dict(zip(FIELDS, ["1", "0", "0", "1", "a", "1", "1", "1"])); x.update(kw); return x
    def test_known_numeric_and_categorical_associations_and_exclusion(self):
        rows = [self.row(**{"MRE-": str(i), "N_estimators": str(i), "criterion": "a" if i < 3 else "b", "ACC+": str(100-i)}) for i in range(1, 7)]
        result = rank(self.make(rows)); by = {x["setting"]: x for x in result["settings"]}
        self.assertAlmostEqual(by["N_estimators"]["score"], 1.0)
        self.assertEqual(result["excluded_outcomes"], ["ACC+", "MRE-", "PRED40+"])
        self.assertEqual(by["criterion"]["groups"]["a"]["count"], 2)
    def test_constants_missing_all_missing_and_determinism(self):
        rows = [self.row(**{"MRE-": "1", "N_estimators": "1", "criterion": "", "Max_depth": ""}), self.row(**{"MRE-": "2", "N_estimators": "1", "criterion": "", "Max_depth": "nan"})]
        a = rank(self.make(rows)); b = rank(self.make(rows)); self.assertEqual(a, b)
        self.assertEqual(a["outcome"]["valid_count"], 2)
        self.assertIn(a["settings"][-1]["score"], (0.0, None))
    def test_nonfinite_outcome_is_excluded(self):
        rows = [self.row(**{"MRE-": "nan", "N_estimators": "1"}), self.row(**{"MRE-": "2", "N_estimators": "2"})]
        self.assertEqual(rank(self.make(rows))["outcome"]["valid_count"], 1)

if __name__ == "__main__": unittest.main()
