import json, math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot import screen, robust_reference

class SpotTests(unittest.TestCase):
    def row(self, i, x): return {"idX": i, "Kloc+": x, "Effort-": x, "Defects-": x}
    def test_unusual_and_ordinary(self):
        r = screen([self.row(i, x) for i, x in enumerate([1, 2, 2, 3, 100])])
        self.assertEqual(r["unique_flagged_record_count"], 1); self.assertEqual(r["per_outcome_flag_counts"]["Kloc+"], 1)
    def test_invalid_and_original_duplicate_ids(self):
        rows = [self.row("same", 1), self.row("same", -1), self.row("other", None)]
        r = screen(rows); self.assertEqual(r["records"][0]["identifier"], "same")
        self.assertEqual(r["records"][1]["row_index"], 2)
    def test_small_empty_constant_zero_scale(self):
        self.assertEqual(screen([])["unique_flagged_record_count"], 0)
        self.assertIsNone(robust_reference([])["median"])
        self.assertEqual(screen([self.row(1, 2), self.row(2, 2)])["unique_flagged_record_count"], 0)
        self.assertIsNone(screen([self.row(1, 2)])["records"][0]["flagged_fields"] if screen([self.row(1, 2)])["records"] else None)
    def test_json_and_arithmetic(self):
        r = screen([self.row(i, x) for i, x in enumerate([1, 2, 3, 4, 100])])
        json.dumps(r, allow_nan=False); ref = robust_reference([1, 2, 3, 4, 100])
        self.assertAlmostEqual(ref["scale"], 1.4826 * statistics_mad([math.log1p(x) for x in [1,2,3,4,100]]))

def statistics_mad(xs):
    import statistics
    m = statistics.median(xs); return statistics.median([abs(x-m) for x in xs])

if __name__ == "__main__": unittest.main()
