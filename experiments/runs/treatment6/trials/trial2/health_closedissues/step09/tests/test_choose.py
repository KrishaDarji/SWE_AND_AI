import csv
import os
import tempfile
import unittest
from choose import choose

FIELDS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth", "MRE-", "ACC+", "PRED40+"]

def make(rows):
    handle = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
    writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows); handle.close()
    return handle.name

def row(n=10, criterion="gini", leaf=1, impurity=0, depth=3, mre=1, acc=1, pred=1):
    return dict(zip(FIELDS, [n, criterion, leaf, impurity, depth, mre, acc, pred]))

class ChooseTests(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, "paths", []): os.unlink(path)
    def data(self, rows):
        path = make(rows); self.paths = getattr(self, "paths", []) + [path]; return path
    def test_cap_and_complete_retention(self):
        path = self.data([row(100, mre=2), row(101, mre=-10), row(99, criterion=" entropy ", leaf=2, impurity=.2, depth=7, mre=1)])
        result = choose(path, 100)
        self.assertEqual(result["selected"]["position"], 4); self.assertEqual(result["selected"]["configuration"]["criterion"], "entropy")
        self.assertEqual(result["over_budget_count"], 1); self.assertEqual(result["valid_count"], 3)
    def test_invalid_and_no_feasible(self):
        path = self.data([row(-1), row(float("nan")), row(10, criterion=""), row(10, mre="bad")])
        result = choose(path, 0); self.assertIsNone(result["selected"]); self.assertEqual(result["invalid_count"], 4)
        with self.assertRaises(ValueError): choose(path, -1)
        with self.assertRaises(ValueError): choose(path, float("inf"))
    def test_each_tiebreaker(self):
        path = self.data([row(10, mre=1, acc=1, pred=1), row(11, mre=1, acc=2, pred=0), row(12, mre=1, acc=2, pred=3), row(13, mre=1, acc=2, pred=3)])
        self.assertEqual(choose(path)["selected"]["position"], 4)
        path = self.data([row(10, mre=1), row(11, mre=0)]); self.assertEqual(choose(path)["selected"]["position"], 3)
    def test_missing_outcome_and_malformed_counts(self):
        path = self.data([row(10), {**row(10), "MRE-": "nan"}])
        with open(path, "a", encoding="utf-8") as handle: handle.write("broken\n")
        result = choose(path); self.assertEqual(result["invalid_count"], 1); self.assertEqual(result["malformed_count"], 1)

if __name__ == "__main__": unittest.main()
