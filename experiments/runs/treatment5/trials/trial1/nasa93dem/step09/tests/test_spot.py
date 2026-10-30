import json, math, os, sys, unittest
HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "source"))
import spot

class SpotTests(unittest.TestCase):
    def rows(self, vals):
        return [{"original_position": i + 1, "idX": str(i), "observed": {"Kloc+": v, "Effort-": v, "Defects-": v},
                 "Kloc+": v, "Effort-": v, "Defects-": v} for i, v in enumerate(vals)]
    def test_unusual_and_ordinary(self):
        out = spot.screen(self.rows([1, 2, 3, 4, 100]))
        self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertEqual(out["per_outcome_flag_counts"]["Kloc+"], 1)
    def test_invalid_and_identifiers(self):
        r = self.rows([1, 2]); r[0]["Kloc+"] = None; r[0]["observed"]["Kloc+"] = None
        out = spot.screen(r)
        self.assertEqual(out["records"][0]["original_position"], 1)
        self.assertEqual(out["records"][0]["idX"], "0")
        json.dumps(out, allow_nan=False)
    def test_duplicate_idx_rows_count_separately(self):
        rows = self.rows([1, 2, 3, 4, 5, 100, 200])
        rows[4]["idX"] = rows[5]["idX"] = "duplicate"
        out = spot.screen(rows)
        self.assertEqual(out["unique_flagged_record_count"], 2)
        self.assertEqual(out["per_outcome_flag_counts"], {"Kloc+": 2, "Effort-": 2, "Defects-": 2})
        self.assertEqual([r["original_position"] for r in out["records"][-2:]], [6, 7])
    def test_small_empty_constant_zero_scale(self):
        self.assertEqual(spot.screen([])["unique_flagged_record_count"], 0)
        out = spot.screen(self.rows([4, 4, 4]))
        self.assertEqual(out["records"][0]["fields"][0]["score"], 0.0)
        out = spot.screen(self.rows([4, 4, 5]))
        self.assertIsNone(out["records"][2]["fields"][0]["score"])
    def test_negative_nonfinite(self):
        self.assertIsNone(spot.finite_nonnegative(-1))
        self.assertIsNone(spot.finite_nonnegative(float("inf")))
        self.assertIsNone(spot.finite_nonnegative("x"))
    def test_score_arithmetic(self):
        c, m, s = spot.reference([0, 1, 3, 7, 15])
        got, _ = spot.score(7, c, m, s)
        self.assertAlmostEqual(got, abs(math.log1p(7)-c)/s)

if __name__ == "__main__": unittest.main()
