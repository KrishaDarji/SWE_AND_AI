import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from justify import analyze, render

def row(i, mre, acc, pred, **kw):
    base = {"record_id": i, "MRE-": mre, "ACC+": acc, "PRED40+": pred, "N_estimators": "40", "criterion": "squared_error", "Min_sample_leaves": "3", "Min_impurity_decrease": "0.1", "Max_depth": "10"}
    base.update(kw); return base

class JustifyTests(unittest.TestCase):
    def test_policy_and_ties(self):
        result = analyze([row(1, .1, .2, .3), row(2, .1, .4, .2), row(3, .1, .4, .5), row(4, .2, .9, .9)])
        self.assertEqual(result["winner"]["record_id"], 3)
    def test_references_and_original_id(self):
        result = analyze([row(7, 1, 2, 3), row(9, 3, 4, 5)])
        self.assertEqual(result["references"]["MRE-"], {"median": 2, "count": 2})
        self.assertIn("record identifier: 7", render(result))
    def test_missing_and_nonfinite(self):
        result = analyze([row(2, "nan", 1, 1), row(8, 1, "", 1), row(11, 2, 2, 2)])
        self.assertEqual(result["eligible_count"], 1); self.assertEqual(result["winner"]["record_id"], 11)
    def test_no_eligible(self):
        result = analyze([row(1, None, 1, 1), row(2, "inf", 1, 1)])
        self.assertIsNone(result["winner"]); self.assertIn("No eligible", render(result))
    def test_all_outcome_medians(self):
        result = analyze([row(1, 1, 10, 100), row(2, 3, 20, 300), row(3, 5, 30, 500)])
        self.assertEqual([result["references"][x]["median"] for x in ("MRE-", "ACC+", "PRED40+")], [3, 20, 300])

if __name__ == "__main__": unittest.main()
