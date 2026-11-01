import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa_summary as n

class TestSpot(unittest.TestCase):
    def row(self, pos, value, ident):
        return {"position": pos, "values": {"idX": ident, "Effort-": value}}

    def test_high_low_self_exclusion_and_counts(self):
        rows = [self.row(i, str(v), "dup") for i, v in enumerate([0, 100, 101, 102, 103, 10000], 2)]
        out = n.spot(rows, threshold=3.5, min_reference=3)
        self.assertEqual(out["valid"], 6); self.assertEqual(out["excluded"], 0)
        self.assertEqual(out["flagged"], 2)
        flagged = {r["position"]: r for r in out["records"] if r["status"] == "flagged"}
        self.assertEqual(set(flagged), {2, 7})
        self.assertEqual(flagged[2]["reference_count"], 5)
        self.assertAlmostEqual(flagged[2]["transformed"], 0.0)
        self.assertEqual(flagged[2]["idX"], flagged[7]["idX"])

    def test_invalid_zero_mad_short_and_boundary(self):
        rows = [self.row(10, "0", "z"), self.row(11, "0", "z"), self.row(12, "0", "z"), self.row(13, "bad", "b"), self.row(14, "-1", "n"), self.row(15, "NaN", "f")]
        out = n.spot(rows, threshold=0, min_reference=2)
        self.assertEqual(out["undefined"], 3); self.assertEqual(out["invalid_or_missing"], 3)
        self.assertTrue(all(r["reason"] == "zero MAD reference scale; score undefined by policy" for r in out["records"][:3]))
        self.assertEqual(out["records"][0]["effort"], 0.0)

    def test_insufficient_reference_and_validation(self):
        out = n.spot([self.row(20, "1", "a"), self.row(21, "2", "b")], min_reference=3)
        self.assertEqual(out["undefined"], 2)
        with self.assertRaises(ValueError): n.spot([], threshold=-1)

if __name__ == "__main__": unittest.main()
