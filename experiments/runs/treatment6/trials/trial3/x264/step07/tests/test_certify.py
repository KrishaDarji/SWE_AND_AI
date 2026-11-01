import math, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class CertifyTests(unittest.TestCase):
    def csv(self, text):
        f=tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); f.write(text); f.close(); self.addCleanup(lambda: Path(f.name).unlink()); return f.name
    def test_groups_direction_variance_and_inference(self):
        r=summary.certify(self.csv("no_8x8dct,Performance-\n0,10\n0,14\n0,12\n1,4\n1,8\n1,6\n"), 5)
        self.assertEqual(r["reference"]["sample_variance"],4); self.assertEqual(r["candidate"]["sample_variance"],4)
        self.assertEqual(r["difference"],-6); self.assertEqual(r["inference"]["standard_error"],math.sqrt(8/3))
        self.assertLess(r["inference"]["degrees_of_freedom"], 5)
        self.assertTrue(r["inference"]["point_meets_practical_threshold"])
    def test_t_reference_values_and_bounds(self):
        self.assertAlmostEqual(summary.t_cdf(0,1),.5)
        self.assertAlmostEqual(summary.t_cdf(1.96,1),.8498, places=3)
        self.assertAlmostEqual(summary.t_cdf(1.96,100000),.975, places=3)
        self.assertAlmostEqual(summary.t_quantile(.975,100000),1.96, places=3)
        self.assertTrue(0 <= summary.t_cdf(-2,3) <= 1)
    def test_invalid_absent_insufficient_and_zero_se(self):
        r=summary.certify(self.csv("no_8x8dct,Performance-\n0,1\n1,2\n")); self.assertFalse(r["inference"]["defined"])
        r=summary.certify(self.csv("no_8x8dct,Performance-\n0,1\n0,1\n1,2\n1,2\n")); self.assertFalse(r["inference"]["defined"])
        r=summary.certify(self.csv("no_8x8dct,Performance-\n0,1\n0,nan\n1,2\n1,3\n2,4\n")); self.assertEqual(r["excluded_count"],2)
        with self.assertRaises(ValueError): summary.certify(self.csv("x,Performance-\n0,1\n"))
        with self.assertRaises(ValueError): summary.certify(self.csv("no_8x8dct,Performance-\n0,1\n1,2\n"), -1)
    def test_threshold_without_significance_is_distinct(self):
        r=summary.certify(self.csv("no_8x8dct,Performance-\n0,10\n0,11\n1,9\n1,10\n"), 0)
        self.assertIn("statistically_significant", r["inference"]); self.assertIsNotNone(r["inference"]["point_meets_practical_threshold"])

if __name__ == "__main__": unittest.main()
