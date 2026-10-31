import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from estimator import assess, DESCRIPTORS

def row(i, effort="", **kw):
    r = {k: "h" for k in DESCRIPTORS}; r.update(idX=str(i), **kw); r["Effort-"] = effort; return r

class FlowTests(unittest.TestCase):
    def test_warmup_and_arithmetic(self):
        out = assess([row(1,"10"),row(2,"20"),row(3,"30"),row(4,"40")])
        self.assertTrue(out[0]["warmup"] and out[1]["warmup"])
        self.assertIsNone(out[1]["prediction_abs_error"])
        self.assertEqual(out[2]["prior_training_count"], 2)
        self.assertAlmostEqual(out[2]["baseline"], 15); self.assertAlmostEqual(out[2]["baseline_abs_error"],15)
        self.assertAlmostEqual(out[3]["cumulative_baseline_mae"], 17.5)
    def test_empty_short_and_invalid_effort(self):
        self.assertEqual(assess([]), [])
        self.assertEqual(len(assess([row(1,"10")])), 1)
        out = assess([row(1,"10"),row(2,"nan"),row(3,"inf"),row(4,"20")])
        self.assertEqual(out[3]["prior_training_count"], 1); self.assertFalse(out[1]["valid_outcome"])
    def test_missing_unseen_and_exclusions(self):
        out = assess([row(1,"10", prec="?"), row(2,"20", prec="new"), row(3,"30", prec="other", **{"Effort-":"30", "Kloc+":"999"})])
        self.assertIsNotNone(out[2]["prediction"]); self.assertEqual(len(DESCRIPTORS), 22)
    def test_freeze_before_reveal_and_future_independence(self):
        base = [row(1,"10"),row(2,"20"),row(3,"30"),row(4,"40")]
        changed = [dict(x) for x in base]; changed[3]["Effort-"]="999"
        self.assertEqual(assess(base)[:3], assess(changed)[:3])
        changed[2]["Effort-"]="bad"; self.assertNotEqual(assess(base)[3]["prior_training_count"], assess(changed)[3]["prior_training_count"])
    def test_prediction_precedes_current_outcome(self):
        a = assess([row(1,"10"),row(2,"20"),row(3,"30")]); b = assess([row(1,"10"),row(2,"20"),row(3,"999")])
        self.assertEqual(a[2]["prediction"], b[2]["prediction"])

if __name__ == "__main__": unittest.main()
