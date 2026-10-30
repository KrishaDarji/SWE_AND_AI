import math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from repair import select

F = dict(N_estimators="10", criterion="squared_error", Min_sample_leaves="2", Min_impurity_decrease="0.5", Max_depth="4")
def r(mre, acc="0.5", pred="0.5", **changes):
    x = dict(F, **changes); x.update({"MRE-": str(mre), "ACC+": acc, "PRED40+": pred}); return x

class RepairTests(unittest.TestCase):
    def test_minimum_change_beats_large_improvement(self):
        z = select([r(0.5, id="base"), r(0.4, N_estimators="11"), r(0.1, N_estimators="11", Max_depth="9")]); self.assertEqual(z["candidate"]["MRE-"], "0.4")
    def test_numeric_equivalence(self): self.assertEqual(select([r(.5), r(.4, N_estimators="10.0")])["change_count"], 0)
    def test_ties_and_tradeoffs(self):
        z = select([r(.5), r(.4, N_estimators="11", **{"ACC+":".9"}), r(.4, N_estimators="12", **{"ACC+":".8", "PRED40+":".9"})]); self.assertEqual(z["candidate"]["N_estimators"], "11")
    def test_reporting_and_worsening(self):
        z = select([r(.5, acc=".9", pred=".4"), r(.4, N_estimators="11", acc=".8", pred=".3")]); self.assertEqual(z["changes"]["N_estimators"], {"old":10.0,"new":11.0}); self.assertTrue(z["secondary"]["ACC+"]["worsened"])
    def test_missing_secondary_invalid_and_none(self):
        self.assertIsNone(select([r(.5, acc=""), r(.4, acc=".6")])["secondary"]["ACC+"]["difference"])
        self.assertEqual(select([r("nan"), r(.4)])["status"], "invalid_baseline")
        self.assertEqual(select([r(.5)])["status"], "no_improvement")
    def test_empty(self): self.assertEqual(select([])["status"], "empty_input")
    def test_missing_criterion_invalidates_baseline(self):
        for missing in (None, "", "   ", "?"):
            self.assertEqual(select([r(.5, criterion=missing), r(.4, N_estimators="11")])["status"], "invalid_baseline")
    def test_missing_criterion_candidate_is_ineligible(self):
        self.assertEqual(select([r(.5), r(.1, criterion="?")])["status"], "no_improvement")
    def test_criterion_whitespace_is_normalized(self):
        z = select([r(.5, criterion=" squared_error "), r(.4, criterion="squared_error")])
        self.assertEqual(z["change_count"], 0)
        self.assertEqual(z["changes"], {})
    def test_valid_criterion_continues_working(self):
        z = select([r(.5), r(.4, criterion="absolute_error")])
        self.assertEqual(z["candidate"]["criterion"], "absolute_error")
        self.assertEqual(z["changes"]["criterion"], {"old": "squared_error", "new": "absolute_error"})

if __name__ == "__main__": unittest.main()
