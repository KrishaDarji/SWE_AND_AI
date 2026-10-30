import json, math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot_screen import screen

class SpotTests(unittest.TestCase):
    def row(self, a, b=None, c=None, i=1): return {"record_id": i, "MRE-": a, "ACC+": b if b is not None else a, "PRED40+": c if c is not None else a}
    def test_unusual_and_ids(self):
        rows = [self.row(x, x, x, i) for i, x in enumerate([1,1,1,1,1,100], 10)]
        out = screen(rows); self.assertEqual(out["unique_flagged_record_count"], 1); self.assertEqual(out["flags"][0]["record_id"], 15)
        self.assertEqual(out["outcome_flag_counts"], {"MRE-":1,"ACC+":1,"PRED40+":1})
    def test_missing_nonfinite_small_empty(self):
        for rows in ([], [self.row(1)], [self.row(None), self.row(float("nan")), self.row(float("inf"))]):
            out = screen(rows); self.assertEqual(out["flags"], []); self.assertEqual(out["unique_flagged_record_count"], 0)
    def test_constant_zero_scale_and_json(self):
        out = screen([self.row(2,i=1), self.row(2,i=2), self.row(2,i=3)])
        self.assertEqual(out["flags"], []); json.dumps(out, allow_nan=False); self.assertTrue(all(v == 0 for v in out["outcome_flag_counts"].values()))
    def test_both_directions(self):
        out = screen([self.row(x, x, x, i) for i,x in enumerate([10,10,10,10,10,0,100],1)])
        self.assertEqual({x["record_id"] for x in out["flags"]}, {6,7})
    def test_score_arithmetic_nonzero_mad(self):
        out = screen([self.row(x, x, x, i) for i,x in enumerate([0,1,2,3,4],1)])
        expected = 0.6744897501960817 * (4-2) / 1
        self.assertAlmostEqual(out["flags"], []) if False else self.assertAlmostEqual((4-2) / out["references"]["MRE-"]["scale"], expected)
    def test_score_arithmetic_iqr_fallback(self):
        out = screen([self.row(x, x, x, i) for i,x in enumerate([0,0,0,1,1],1)])
        self.assertEqual(out["references"]["MRE-"]["method"], "median/IQR fallback")
        self.assertAlmostEqual((1-0) / (1/1.349), 1 / out["references"]["MRE-"]["scale"])
    def test_boundary_changes_under_double_scaling(self):
        out = screen([self.row(x, x, x, i) for i,x in enumerate([0,0,0,0,1,1,3],1)])
        self.assertTrue(any(x["record_id"] == 7 and x["outcome"] == "MRE-" for x in out["flags"]))
        corrected = 3 / (1/1.349); erroneous = 0.6744897501960817 * corrected
        self.assertGreater(corrected, 3.5); self.assertLess(erroneous, 3.5)

if __name__ == "__main__": unittest.main()
