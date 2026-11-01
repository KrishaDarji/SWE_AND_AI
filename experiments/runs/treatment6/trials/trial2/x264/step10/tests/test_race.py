import os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import race, _classify_difference

class RaceTests(unittest.TestCase):
    def case(self, text):
        h = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        h.write(text); h.close(); self.addCleanup(lambda: os.unlink(h.name)); return h.name
    def test_equivalence_reproducibility_duplicates_and_charging(self):
        p = self.case("a,Performance-\n0,4\n0,99\n1,bad\n2,1\n?,2\n")
        a = race(p, 20, [42], 0); b = race(p, 20, [42], 0, audit=True)
        self.assertEqual(a["rows"], b["rows"]); self.assertEqual(a["candidate_count"], 2)
        row = a["rows"][0]; self.assertEqual(row["selector"]["attempted_evaluations"], 2)
        self.assertEqual(row["selector"]["attempted_evaluations"], row["baseline"]["attempted_evaluations"])
        self.assertTrue(a["equivalent_methods"])
    def test_difference_boundaries_and_summary(self):
        self.assertEqual([_classify_difference(x, .1) for x in (1, .1, -.1, -.11)], ["win", "tie", "tie", "loss"])
        p = self.case("a,Performance-\n0,4\n1,1\n2,3\n")
        r = race(p, 0, [1, 2], 0); self.assertEqual(r["comparable_pairs"], 0); self.assertIsNone(r["mean_difference"])
        self.assertEqual(r["counts"], {"win": 0, "tie": 0, "loss": 0})
    def test_invalid_inputs_and_empty_pool(self):
        p = self.case("a,Performance-\n?,bad\n")
        self.assertEqual(race(p, 5, [42])["status"], "empty configuration pool")
        with self.assertRaises(ValueError): race(p, -1)
        with self.assertRaises(ValueError): race(p, 1, tolerance=-1)

if __name__ == "__main__": unittest.main()
