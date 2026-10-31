import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "source"))
from flow import StreamEvaluator, evaluate_csv


class FlowTests(unittest.TestCase):
    def test_empty_and_one(self):
        e = StreamEvaluator()
        self.assertEqual(e.final()["MRE-"]["count"], 0)
        row = {"MRE-": "2", "ACC+": "0.5", "PRED40+": "?", "N_estimators": "1"}
        e.receive(row, 7)
        self.assertEqual(e.final()["MRE-"]["sample_variance"], 0.0)
        self.assertEqual(e.final()["PRED40+"]["count"], 0)

    def test_batch_statistics_directions_best_and_tie(self):
        rows = [
            {"MRE-": "3", "ACC+": "1", "PRED40+": "0", "N_estimators": "a"},
            {"MRE-": "1", "ACC+": "2", "PRED40+": "4", "N_estimators": "b"},
            {"MRE-": "1", "ACC+": "2", "PRED40+": "2", "N_estimators": "c"},
        ]
        e = StreamEvaluator()
        for i, row in enumerate(rows, 1): e.receive(row, i)
        expected = {"MRE-": (3, 5 / 3, 1), "ACC+": (3, 5 / 3, 2), "PRED40+": (3, 2, 4)}
        for key, (n, mean, best) in expected.items():
            got = e.final()[key]
            self.assertEqual(got["count"], n)
            self.assertAlmostEqual(got["mean"], mean)
            self.assertEqual(got["best"], best)
        self.assertEqual(e.final()["MRE-"]["best_record_id"], 2)
        self.assertEqual(e.final()["ACC+"]["best_record_id"], 2)

    def test_missing_nonfinite_and_defensive_snapshots(self):
        e = StreamEvaluator()
        row = {"MRE-": "1", "ACC+": "nan", "PRED40+": "2", "N_estimators": "x"}
        first = e.receive(row, 1)
        row["N_estimators"] = "mutated"
        e.receive({"MRE-": "2", "ACC+": "inf", "PRED40+": "3", "N_estimators": "y"}, 2)
        self.assertEqual(first["outcomes"]["MRE-"]["best_configuration"]["N_estimators"], "x")
        self.assertEqual(e.final()["ACC+"]["count"], 0)

    def test_iterator_consumption(self):
        consumed = []
        def source():
            for n in range(3):
                consumed.append(n)
                yield {"MRE-": str(n), "ACC+": str(n), "PRED40+": str(n)}
        e = StreamEvaluator()
        for i, row in enumerate(source(), 1): e.receive(row, i)
        self.assertEqual(consumed, [0, 1, 2])
        self.assertEqual(len(e.snapshots), 3)

    def test_csv_integration(self):
        fd, path = tempfile.mkstemp(text=True)
        try:
            with os.fdopen(fd, "w") as f:
                f.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n")
                f.write("1,2,3,4,c,5,6,7\n")
            e = evaluate_csv(path)
            self.assertEqual(e.received, 1)
            self.assertEqual(e.final()["ACC+"]["best_configuration"]["criterion"], "c")
        finally:
            os.unlink(path)


if __name__ == "__main__": unittest.main()
