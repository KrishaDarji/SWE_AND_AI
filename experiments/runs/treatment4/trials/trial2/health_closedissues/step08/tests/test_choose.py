import csv
import math
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from choose import choose

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class ChooseTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name

    def row(self, **changes):
        base = {"MRE-": "9", "ACC+": ".1", "PRED40+": ".1", "N_estimators": "10", "criterion": "x", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"}
        base.update({k: str(v) for k, v in changes.items()}); return base

    def test_cap_boundaries_and_best_feasible(self):
        result = choose(self.write([self.row(N_estimators=100, **{"MRE-": 1}), self.row(N_estimators=101, **{"MRE-": 0})]), 100)
        self.assertEqual(result["feasible_count"], 1); self.assertEqual(result["selected"]["record_id"], 1)
        self.assertEqual(result["total_count"], 2)

    def test_all_tiebreakers_and_earliest(self):
        rows = [self.row(**{"MRE-": 1, "ACC+": .5, "PRED40+": .5}), self.row(**{"MRE-": 1, "ACC+": .6, "PRED40+": .1}), self.row(**{"MRE-": 1, "ACC+": .6, "PRED40+": .9}), self.row(**{"MRE-": 1, "ACC+": .6, "PRED40+": .9})]
        self.assertEqual(choose(self.write(rows))["selected"]["record_id"], 3)

    def test_invalid_caps_and_bad_rows(self):
        path = self.write([self.row(**{"MRE-": "nan"}), self.row(N_estimators="bad")])
        for cap in (-1, float("nan"), "bad"):
            with self.assertRaises(ValueError): choose(path, cap)
        with self.assertRaises(ValueError): choose(path, 100)

    def test_empty_and_no_feasible(self):
        empty = self.write([])
        with self.assertRaises(ValueError): choose(empty)
        with self.assertRaises(ValueError): choose(self.write([self.row(N_estimators=101)]), 100)

if __name__ == "__main__": unittest.main()
