import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from certify import certify, load_groups, summary, welch, t_cdf, t_quantile

class CertifyTests(unittest.TestCase):
    def test_group_selection_and_whitespace(self):
        path = os.path.join(os.path.dirname(__file__), "fixture.csv")
        groups = load_groups(path)
        self.assertEqual(groups, {"n": [1.0, 3.0], "h": [2.0, 4.0]})
    def test_direction_b_and_a(self):
        self.assertEqual(certify({"n":[10,11,9], "h":[1,2,3]}, 1)["direction"], "favors B (tool=h)")
        self.assertEqual(certify({"n":[1,2,3], "h":[10,11,9]}, 1)["direction"], "favors A (tool=n)")
    def test_welch_unequal_variance_reference(self):
        r = welch([1,2,3], [10,20,30,40])
        self.assertAlmostEqual(r["df"], 3.0478994017, places=8)
        self.assertAlmostEqual(r["ci"][0], -43.44245, delta=0.02)
    def test_known_t_values(self):
        self.assertAlmostEqual(t_cdf(0, 10), .5, places=12)
        self.assertAlmostEqual(t_quantile(.975, 10), 2.2281388519, places=7)
    def test_insufficient_missing_nonfinite(self):
        self.assertEqual(summary([])["n"], 0)
        with self.assertRaises(ValueError): certify({"n":[1], "h":[2,3]}, 0)
        with self.assertRaises(ValueError): certify({"n":[1,2], "h":[2,3]}, -1)
    def test_zero_variance(self):
        r = certify({"n":[2,2,2], "h":[2,2,2]}, 0)
        self.assertEqual(r["p"], 1.0)
    def test_threshold_point_crosses_interval(self):
        r = certify({"n":[10,11,9,10], "h":[1,2,3,2]}, 7)
        self.assertTrue(r["point_exceeds"])
        self.assertFalse(r["interval_above"])
    def test_threshold_below_interval_and_no_significance_requirement(self):
        r = certify({"n":[2,2,2], "h":[1,1,1]}, .5)
        self.assertTrue(r["interval_above"])

if __name__ == "__main__": unittest.main()
