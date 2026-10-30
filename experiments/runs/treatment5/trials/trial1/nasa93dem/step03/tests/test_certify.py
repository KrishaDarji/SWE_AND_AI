import math, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import compare, load_groups, t_cdf, t_quantile

class TestCertify(unittest.TestCase):
    def test_group_selection_and_whitespace(self):
        p = Path(__file__).parents[1] / "fixture.csv"; p.write_text("tool,Effort-\n n , 1 \n h,2\n x,9\n n,3\n h,4\n")
        self.assertEqual(load_groups(p), {"n":[1.,3.], "h":[2.,4.]})
    def test_direction_and_threshold_crossing(self):
        r=compare([1,2,3,4],[10,11,12,13], 8); self.assertLess(r["difference"],0); self.assertTrue(r["point_exceeds"]); self.assertFalse(r["entirely_practical"])
        r=compare([10,11,12,13],[1,2,3,4], 5); self.assertGreater(r["difference"],0); self.assertTrue(r["point_exceeds"])
    def test_welch_known_values(self):
        r=compare([1,2,3],[4,5,100],0); self.assertAlmostEqual(r["df"],2.001315645, places=6); self.assertAlmostEqual(r["difference"],-34.333333, places=5)
    def test_reference_t_values(self):
        for t in (.001, 2, 10000, -.001, -2, -10000):
            self.assertAlmostEqual(t_cdf(t, 1), .5 + math.atan(t)/math.pi, places=10 if abs(t)<100 else 8)
            self.assertGreaterEqual(t_cdf(t, 1), 0); self.assertLessEqual(t_cdf(t, 1), 1)
        self.assertAlmostEqual(t_cdf(2, 10), 1-t_cdf(-2,10), places=12)
        self.assertAlmostEqual(t_quantile(.975, 10), 2.22813885, places=5)
        for p in (0, 1, float('nan')):
            with self.assertRaises(ValueError): t_quantile(p, 10)
    def test_invalid_and_insufficient(self):
        for x in (-1, float("nan")): 
            with self.assertRaises(ValueError): compare([1,2],[2,3],x)
        with self.assertRaises(ValueError): compare([1],[2,3],0)
    def test_zero_variance_missing_nonfinite(self):
        r=compare([2,2,2],[2,2,2],0); self.assertEqual(r["status"], "undefined"); self.assertIsNone(r["p_two_sided"])
        r=compare([2,2,2],[3,3,3],0); self.assertEqual(r["status"], "undefined"); self.assertIsNone(r["ci_low"])
        r=compare([2,2,2],[1,2,4],0); self.assertNotEqual(r.get("status"), "undefined"); self.assertIsNotNone(r["p_two_sided"])
        self.assertEqual(load_groups(Path(__file__).parents[1]/"fixture.csv"), {"n":[1.,3.],"h":[2.,4.]})

if __name__ == "__main__": unittest.main()
