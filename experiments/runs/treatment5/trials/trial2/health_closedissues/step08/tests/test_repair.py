import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from repair import choose

F = {"N_estimators": 40, "criterion": "squared_error", "Min_sample_leaves": 3,
     "Min_impurity_decrease": 2.75, "Max_depth": 10}
def r(i, mre, acc=.5, pred=.5, **kw):
    x = {"identifier": i, "MRE-": mre, "ACC+": acc, "PRED40+": pred, **F}
    x.update(kw); return x

class RepairTests(unittest.TestCase):
    def test_min_change_beats_big_improvement(self):
        out = choose(r(1, .5), [r(2, .49, N_estimators=41), r(3, .1, N_estimators=41, Max_depth=9)])
        self.assertEqual(out["candidate_identifier"], 2)
    def test_numeric_equivalence(self):
        out = choose(r(1, .5, Min_impurity_decrease=2), [r(2, .4, Min_impurity_decrease=2.0)])
        self.assertEqual(out["change_count"], 0)
    def test_ties(self):
        out = choose(r(1, .5), [r(3, .4, acc=.7, pred=.1, N_estimators=41), r(2, .4, acc=.7, pred=.1, N_estimators=41)])
        self.assertEqual(out["candidate_identifier"], 2)
    def test_all_tiebreakers(self):
        out = choose(r(1, .5), [r(2, .4, acc=.7, pred=.2, N_estimators=41), r(3, .4, acc=.8, pred=.1, N_estimators=41)])
        self.assertEqual(out["candidate_identifier"], 3)
    def test_changes_and_tradeoff(self):
        out = choose(r(1, .5, acc=.8, pred=.8), [r(2, .4, acc=.7, pred=.9, N_estimators=41)])
        self.assertEqual(out["changes"][0]["old"], 40); self.assertTrue(out["ACC+_worsened"])
    def test_missing_secondary(self):
        out = choose(r(1, .5, acc=None), [r(2, .4)])
        self.assertIsNone(out["ACC+_difference"]); self.assertIsNone(out["ACC+_worsened"])
    def test_invalid_and_no_improvement(self):
        self.assertEqual(choose(r(1, math.nan), [r(2, .1)])["status"], "invalid_baseline")
        self.assertEqual(choose(r(1, .5), [r(2, .5)])["status"], "no_improvement")
    def test_empty(self):
        self.assertEqual(choose(r(1, .5), [])["status"], "no_improvement")

if __name__ == "__main__": unittest.main()
