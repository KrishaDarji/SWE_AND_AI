import os,tempfile,unittest
from source.certify import certify,t_cdf,t_critical
H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def make(body):
 f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.write(H+body); f.close(); return f.name
class CertifyTests(unittest.TestCase):
 def test_fixed_groups_direction_and_welch(self):
  p=make("1,2,3,1,squared_error,1,0,1\n3,4,5,2,squared_error,1,0,1\n2,8,9,3,absolute_error,1,0,1\n6,7,8,4,absolute_error,1,0,1\n"); self.addCleanup(lambda:os.unlink(p)); x=certify(p); self.assertEqual(x["comparison"]["candidate"],"absolute_error"); self.assertEqual(x["comparison"]["difference_candidate_minus_reference"],2); self.assertAlmostEqual(x["reference"]["sample_variance"],2); self.assertAlmostEqual(x["inference"]["degrees_of_freedom"],1.4705882352941178)
 def test_tail_and_threshold_and_secondary_independence(self):
  self.assertAlmostEqual(t_cdf(0,5),.5); self.assertAlmostEqual(t_critical(1),12.7062047,5); self.assertTrue(0<=t_cdf(100,10)<=1)
  p=make("1,1,1,1,squared_error,1,0,1\n2,2,2,2,squared_error,1,0,1\n0,3,4,3,absolute_error,1,0,1\n0,5,6,4,absolute_error,1,0,1\n"); self.addCleanup(lambda:os.unlink(p)); x=certify(p,.5); self.assertEqual(x["secondary"]["ACC+"]["squared_error"]["finite_count"],2); self.assertEqual(x["secondary"]["PRED40+"]["absolute_error"]["direction"],"maximize"); self.assertTrue(x["threshold_interpretation"]["point_estimate_meets"])
 def test_missing_absent_insufficient_and_zero(self):
  p=make("1,?,nan,1,squared_error,1,0,1\n2,3,4,2,absolute_error,1,0,1\n"); self.addCleanup(lambda:os.unlink(p)); self.assertEqual(certify(p)["inference"]["status"],"undefined")
  with self.assertRaises(ValueError): certify(p,-1)
if __name__=="__main__": unittest.main()
