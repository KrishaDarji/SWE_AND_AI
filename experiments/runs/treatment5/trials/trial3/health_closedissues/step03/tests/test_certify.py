import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..','source'))
import certify
class T(unittest.TestCase):
 def test_groups_direction(self):
  a=certify.assess({'squared_error':[1,2,3],'absolute_error':[.5,1,1.5]}); self.assertLess(a['welch']['difference'],0)
 def test_welch_reference(self):
  w=certify.welch([1,2,3],[4,5,6]); self.assertAlmostEqual(w['df'],4,12); self.assertAlmostEqual(w['ci'][0],.733042,5)
 def test_threshold_uncertain(self):
  a=certify.assess({'squared_error':[1,2,3],'absolute_error':[.5,1,1.5]},.6); self.assertEqual(a['practical_importance'],'point improvement beyond threshold')
 def test_missing_insufficient_zero(self):
  self.assertEqual(certify.assess({'squared_error':[1],'absolute_error':[2]})['welch'],None)
  self.assertIsNone(certify.welch([1,1],[2,2])['ci'])
 def test_beta_and_t_known(self):
  self.assertAlmostEqual(certify.beta_cdf(.5,1,1),.5,12); self.assertAlmostEqual(certify.t_quantile(.975,4),2.776445,5); self.assertTrue(0<=certify.t_cdf(2,4)<=1)
 def test_invalid_threshold(self):
  with self.assertRaises(ValueError): certify.assess({'squared_error':[1,2],'absolute_error':[2,3]},-1)
if __name__=='__main__': unittest.main()
