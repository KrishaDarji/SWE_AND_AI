import math, pathlib, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa93_summary as ns

def make(values):
    f = tempfile.NamedTemporaryFile("w", delete=False, newline="")
    f.write("idX,Effort-,prec\n" + "\n".join(f"{i},{v},h" for i, v in enumerate(values, 1)) + "\n"); f.close(); return f.name

class SpotTests(unittest.TestCase):
    def test_leave_one_out_high_low_and_counts(self):
        r = ns.spot(make([1, 1, 1, 100]), threshold=3.5, min_reference=3)
        self.assertEqual(r["reports"][-1]["status"], "undefined")
        r = ns.spot(make([1, 2, 3, 4, 100]), threshold=1, min_reference=3)
        self.assertEqual(r["reports"][-1]["status"], "flagged")
        self.assertEqual(r["counts"]["total_valid"] + r["counts"]["excluded"], 5)
    def test_invalid_negative_zero_and_undefined(self):
        r = ns.spot(make([0, -1, "", "bad", "NaN"]), min_reference=1)
        self.assertEqual(r["counts"]["excluded"], 4); self.assertEqual(r["reports"][0]["observed_effort"], 0)
        self.assertEqual(r["reports"][1]["reason"], "negative Effort- cannot be transformed by log1p")
    def test_duplicates_positions_and_zero_mad(self):
        r = ns.spot(make([2, 2, 2, 2]), min_reference=1)
        self.assertTrue(all(x["status"] == "undefined" for x in r["reports"]))
        self.assertEqual([x["position"] for x in r["reports"]], [2, 3, 4, 5])
        self.assertEqual([x["idX"] for x in r["reports"]], ["1", "2", "3", "4"])
    def test_independent_median_mad(self):
        r = ns.spot(make([0, 1, 2, 10]), threshold=100, min_reference=3)
        row = r["reports"][2]; refs = [math.log1p(x) for x in [0, 1, 10]]
        med = refs[1]; mad = ns._median([abs(x-med) for x in refs])
        self.assertAlmostEqual(row["center"], med); self.assertAlmostEqual(row["scale"], mad)

if __name__ == "__main__": unittest.main()
