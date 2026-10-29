import math
import unittest
from source.nasa93_flow import assess_arrivals

COLS = ["idX", "a", "b", "Effort-", "Kloc+", "Defects-"]

def rows(values):
    return [{"idX": i + 1, "a": a, "b": b, "Effort-": effort, "Kloc+": 1, "Defects-": 1}
            for i, (a, b, effort) in enumerate(values)]

class FlowTests(unittest.TestCase):
    def test_warmup_and_cumulative_arithmetic(self):
        out = assess_arrivals(rows([("a", "x", 10), ("b", "x", 20), ("c", "y", 30), ("d", "y", 40)]), COLS)
        self.assertEqual([x["status"] for x in out[:2]], ["warm-up", "warm-up"])
        self.assertEqual(out[2]["prior_training_count"], 2)
        self.assertAlmostEqual(out[2]["baseline"], 15)
        self.assertAlmostEqual(out[2]["baseline_cumulative_mae"], 15)
        self.assertTrue(math.isfinite(out[3]["cumulative_mae"]))

    def test_empty_short_and_invalid_effort(self):
        self.assertEqual(assess_arrivals([], COLS), [])
        self.assertEqual(len(assess_arrivals(rows([("a", "x", 1)]), COLS)), 1)
        out = assess_arrivals(rows([("a", "x", 1), ("b", "x", 2), ("c", "x", float("nan")), ("d", "x", 4)]), COLS)
        self.assertEqual(out[2]["status"], "invalid_effort_not_revealed")
        self.assertEqual(out[3]["prior_training_count"], 2)

    def test_future_outcomes_and_records_do_not_change_current_prediction(self):
        base = rows([("a", "x", 10), ("b", "x", 20), ("c", "y", 30), ("d", "y", 40)])
        changed = rows([("a", "x", 10), ("b", "x", 20), ("c", "y", 999), ("d", "y", 888), ("e", "z", 777)])
        a, b = assess_arrivals(base, COLS), assess_arrivals(changed, COLS)
        self.assertEqual(a[2]["prediction"], b[2]["prediction"])
        self.assertEqual(a[2]["baseline"], b[2]["baseline"])
        self.assertEqual(a[2]["prior_training_count"], b[2]["prior_training_count"])

    def test_missing_and_unseen_descriptors(self):
        out = assess_arrivals(rows([("a", None, 10), (None, "x", 20), ("new", "unseen", 30)]), COLS)
        self.assertTrue(math.isfinite(out[2]["prediction"]))

if __name__ == "__main__": unittest.main()
