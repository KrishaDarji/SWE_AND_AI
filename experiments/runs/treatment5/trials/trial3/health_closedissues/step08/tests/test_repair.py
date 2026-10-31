import math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from repair import choose

FIELDS = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
def row(i, mre, acc, pred, **kw):
    d = {"identifier": i, "MRE-": mre, "ACC+": acc, "PRED40+": pred, "N_estimators": 10, "criterion": "squared_error", "Min_sample_leaves": 2, "Min_impurity_decrease": 0.5, "Max_depth": 5}; d.update(kw); return d

class Tests(unittest.TestCase):
    def test_minimum_change_beats_improvement(self):
        r = choose([row(1, .8, .5, .5), row(2, .7, .1, .1, Max_depth=6), row(3, .1, .9, .9, Max_depth=6, N_estimators=20)])
        self.assertEqual(r["candidate"]["identifier"], 2)
    def test_numeric_equivalence(self):
        r = choose([row(1, .8, .5, .5), row(2, .7, .5, .5, N_estimators="10.0")]); self.assertEqual(r["change_count"], 0)
    def test_tiebreakers(self):
        r = choose([row(1,.8,.5,.5), row(2,.6,.6,.1,Max_depth=6), row(3,.6,.7,.9,Max_depth=7), row(4,.6,.7,.9,Max_depth=8)])
        self.assertEqual(r["candidate"]["identifier"], 3)
    def test_exact_changes_and_tradeoff(self):
        r = choose([row(1,.8,.5,.5), row(2,.7,.4,.4,Max_depth=6)]); self.assertEqual(r["changes"], [{"field":"Max_depth","old":5,"new":6}]); self.assertTrue(r["ACC+_worsened"])
    def test_missing_secondary(self):
        r = choose([row(1,.8,"",.5), row(2,.7,.6,.6,Max_depth=6)])
        self.assertIsNone(r["ACC+_difference"]); self.assertIsNone(r["ACC+_worsened"])
    def test_invalid_baseline(self):
        with self.assertRaisesRegex(ValueError, "invalid baseline"): choose([row(1,"nan",.5,.5)])
    def test_no_improvement_empty(self):
        self.assertEqual(choose([row(1,.5,.5,.5), row(2,.6,.5,.5)])['status'], 'no_improvement')
        with self.assertRaises(ValueError): choose([])

if __name__ == "__main__": unittest.main()
