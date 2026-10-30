import math, os, sys, unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','source'))
import certify

class T(unittest.TestCase):
 def test_groups_direction(self):
  r=certify.welch([1,2,3],[0,1,2]); self.assertAlmostEqual(r['difference'],-1); self.assertLess(r['p'],1)
 def test_welch_known(self):
  r=certify.welch([1,2,3,4],[2,4,6,8]); self.assertAlmostEqual(r['df'],4.411764705882353,12); self.assertTrue(0<=r['p']<=1); self.assertLess(r['ci_low'],r['ci_high'])
 def test_cdf_reference(self):
  self.assertAlmostEqual(certify.student_cdf(0,1),.5,12); self.assertAlmostEqual(certify.student_cdf(12,10),.99999999,5); self.assertTrue(0<=certify.student_cdf(-100,3)<=1)
 def test_threshold(self):
  with self.assertRaises(ValueError): certify.summarize('/dev/null',-1)
  self.assertEqual(certify.summarize('/dev/null')['practical_importance'],'undetermined (no threshold supplied)')
 def test_missing_insufficient_zero(self):
  self.assertIsNone(certify.welch([1],[2,3])); self.assertIsNone(certify.welch([1,1],[1,1]))
if __name__=='__main__': unittest.main()
