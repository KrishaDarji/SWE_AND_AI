import csv
import io
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from flow import FlowEvaluator, evaluate_csv


def row(a, b, c, n=10, criterion="squared_error", leaf=1, impurity=0.25, depth=3):
    return {"MRE-": str(a), "ACC+": str(b), "PRED40+": str(c),
            "N_estimators": str(n), "criterion": criterion,
            "Min_sample_leaves": str(leaf), "Min_impurity_decrease": str(impurity),
            "Max_depth": str(depth)}


class FlowTests(unittest.TestCase):
    def test_empty_and_one(self):
        e = FlowEvaluator(); self.assertEqual(e.summary()["MRE-"]["count"], 0)
        e.accept(row(2, 3, 4)); s = e.summary()
        self.assertEqual(s["MRE-"]["mean"], 2.0); self.assertIsNone(s["MRE-"]["sample_variance"])

    def test_batch_stats_directions_tie_and_copy(self):
        e = FlowEvaluator(); first = row(2, 1, 5, n=20); e.accept(first)
        e.accept(row(2, 3, 4, n=30)); e.accept(row(1, 2, 6, n=40))
        s = e.summary()
        self.assertEqual(s["MRE-"]["best"]["record_id"], 3)
        self.assertEqual(s["ACC+"]["best"]["record_id"], 2)
        self.assertEqual(s["PRED40+"]["best"]["record_id"], 3)
        self.assertAlmostEqual(s["MRE-"]["sample_variance"], 1 / 3)
        old = e.snapshots[0]; old["MRE-"]["best"]["configuration"]["N_estimators"] = 999
        self.assertEqual(e.snapshots[0]["MRE-"]["best"]["configuration"]["N_estimators"], 20)

    def test_missing_nonfinite_and_iterator_csv(self):
        e = FlowEvaluator(); r = row(1, 2, 3); r["ACC+"] = "nan"; r["PRED40+"] = ""
        e.accept(r); self.assertEqual(e.summary()["ACC+"]["count"], 0)
        text = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n1,2,3,10,squared_error,1,0.25,3\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(text); name = f.name
        try: self.assertEqual(len(evaluate_csv(name).snapshots), 1)
        finally: os.unlink(name)


if __name__ == "__main__": unittest.main()
