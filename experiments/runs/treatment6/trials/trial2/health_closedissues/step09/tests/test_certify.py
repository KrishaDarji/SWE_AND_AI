import csv, math, os, tempfile, unittest
import sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../source"))
from certify import certify, t_cdf, t_quantile

H = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def fixture(body):
    f = tempfile.NamedTemporaryFile(mode="w", delete=False); f.write(H + body); f.close(); return f.name

class CertifyTests(unittest.TestCase):
    def test_fixed_groups_difference_variance_and_welch(self):
        p = fixture("1,10,20,1,squared_error,1,0,1\n3,20,30,2,squared_error,1,0,1\n2,30,40,1, absolute_error ,1,0,1\n4,40,50,2,absolute_error,1,0,1\n")
        r = certify(p, .5); self.assertEqual(r["comparison"]["groups"], {"candidate": 2, "reference": 2})
        self.assertAlmostEqual(r["primary"]["difference"], 1); self.assertAlmostEqual(r["primary"]["candidate_variance"], 2); self.assertAlmostEqual(r["primary"]["reference_variance"], 2)
        self.assertAlmostEqual(r["primary"]["standard_error"], math.sqrt(2)); self.assertAlmostEqual(r["primary"]["df"], 2); self.assertEqual(r["practical"], {"point_meets": False, "ci_supports": False, "threshold": .5}); os.unlink(p)
    def test_t_references_and_probability_bounds(self):
        self.assertAlmostEqual(t_cdf(0, 10), .5); self.assertTrue(0 <= t_cdf(100, 10) <= 1)
        self.assertAlmostEqual(t_quantile(.975, 1), 12.7062, places=3); self.assertAlmostEqual(t_quantile(.975, 10), 2.2281, places=3); self.assertAlmostEqual(t_quantile(.975, 30), 2.0423, places=3)
    def test_missing_absent_insufficient_and_zero_se(self):
        p = fixture("1,1,1,1,squared_error,1,0,1\nNA,2,2,2,squared_error,1,0,1\n3,3,3,3,absolute_error,1,0,1\n")
        r = certify(p); self.assertFalse(r["primary"]["defined"]); self.assertEqual(r["primary"]["reason"], "each group needs at least two finite outcomes"); os.unlink(p)
        p = fixture("1,1,1,1,squared_error,1,0,1\n1,2,2,2,squared_error,1,0,1\n1,3,3,3,absolute_error,1,0,1\n1,3,3,4,absolute_error,1,0,1\n")
        r = certify(p); self.assertFalse(r["primary"]["defined"]); self.assertIn("zero standard error", r["primary"]["reason"]); os.unlink(p)
    def test_threshold_validation_and_independent_secondary_counts(self):
        p = fixture("1,1,1,1,squared_error,1,0,1\n2,2,,2,squared_error,1,0,1\n0,3,4,3,absolute_error,1,0,1\n0,NA,5,4,absolute_error,1,0,1\n")
        r = certify(p, 1); self.assertEqual(r["secondary"]["ACC+"]["candidate"]["finite_count"], 1); self.assertEqual(r["secondary"]["PRED40+"]["candidate"]["finite_count"], 2)
        with self.assertRaises(ValueError): certify(p, -1)
        os.unlink(p)
if __name__ == "__main__": unittest.main()
