import math, pathlib, sys, unittest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from justify import choose, medians, render

def rec(i, mre, acc, pred, **settings):
    return {"record_id": i, "outcomes": {"MRE-": mre, "ACC+": acc, "PRED40+": pred}, "settings": settings}

class JustifyTests(unittest.TestCase):
    def test_selection_and_tiebreakers(self):
        rows = [rec(3, .1, .2, .9), rec(2, .1, .3, .1), rec(1, .1, .3, .2)]
        self.assertEqual(choose(rows)[0]["record_id"], 1)
    def test_medians_and_counts_ignore_invalid_per_outcome(self):
        rows = [rec(1, 1, 2, float("nan")), rec(2, 3, "bad", 4), rec(3, 5, 6, 8)]
        got = medians(rows)
        self.assertEqual(got["MRE-"]["median"], 3)
        self.assertEqual(got["MRE-"]["valid_count"], 3)
        self.assertEqual(got["ACC+"]["median"], 4)
        self.assertEqual(got["ACC+"]["valid_count"], 2)
        self.assertEqual(got["PRED40+"]["median"], 6)
        self.assertEqual(got["PRED40+"]["valid_count"], 2)
    def test_evidence_labels_and_original_id_in_render(self):
        text = render([rec(9, 1, 2, 3, N_estimators="10")])
        self.assertIn("identifier: 9", text.lower())
        self.assertIn("evidence_label=measured", text)
    def test_no_eligible_records(self):
        text = render([rec(1, None, 2, 3), rec(2, 1, math.inf, 3)])
        self.assertIn("No eligible records", text)

if __name__ == "__main__": unittest.main()
