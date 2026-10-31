import json, math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot import screen

def row(i, k="10", e="20", d="30"): return {"idX": i, "Kloc+": k, "Effort-": e, "Defects-": d}

class SpotTests(unittest.TestCase):
    def test_arithmetic_and_flags(self):
        out = screen([row(str(i), str(10 + i), "20", "30") for i in range(5)] + [row("C", "10000", "20", "30")])
        expected = abs(math.log1p(10000)-out["references"]["Kloc+"]["median"]) / out["references"]["Kloc+"]["iqr"]
        self.assertAlmostEqual(out["records"][5]["flags"][0]["score"], expected, places=12)
        self.assertEqual(out["records"][5]["identifier"], "C")
        self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertEqual(out["per_outcome_flag_counts"]["Kloc+"], 1)
    def test_invalid_and_negative(self):
        out = screen([row("x", "", "NaN", "-2")])
        reasons = {f["reason"] for f in out["records"][0]["flags"]}
        self.assertEqual(reasons, {"missing_or_nonfinite", "negative_value"})
    def test_small_empty_and_constant(self):
        self.assertEqual(screen([])["unique_flagged_record_count"], 0)
        self.assertEqual(screen([row("z", "5", "5", "5")])["per_outcome_flag_counts"], {"Kloc+": 0, "Effort-": 0, "Defects-": 0})
        out = screen([row("a", "5", "5", "5")])
        self.assertEqual(out["unique_flagged_record_count"], 0)
    def test_strict_json(self):
        json.dumps(screen([row("x", "inf", "2", "3")]), allow_nan=False)

if __name__ == "__main__": unittest.main()
