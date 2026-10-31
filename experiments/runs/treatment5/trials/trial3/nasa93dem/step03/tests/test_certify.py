import math,sys,unittest
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__),'../source'))
from certify import compare,mean_var,t_cdf
class T(unittest.TestCase):
 def test_groups_and_whitespace(self): self.assertEqual(mean_var([1,3])[1],2)
 def test_direction_and_threshold_crossing(self):
  r=compare([10,11,12,13],[20,21,22,23],8); self.assertLess(r['difference'],0); self.assertTrue(r['magnitude_exceeds']); self.assertFalse(r['entirely_practical'])
 def test_unequal_welch(self):
  r=compare([1,2,3],[10,10.1,10.2,10.3,10.4],0); self.assertGreater(r['df'],2); self.assertLess(r['ci_low'],r['difference'])
 def test_invalid_and_insufficient(self):
  with self.assertRaises(ValueError):compare([1],[2,3],0)
  with self.assertRaises(ValueError):compare([1,2],[3,4],-1)
 def test_zero_missing_nonfinite(self):
  r=compare([1,1,math.nan],[1,1,math.inf],0); self.assertEqual(r['p_value'],1.0); self.assertEqual(r['ci_low'],0)
 def test_reference_t(self): self.assertAlmostEqual(t_cdf(0,5),.5,places=12); self.assertAlmostEqual(t_cdf(2.570582,5),.975,places=5)
if __name__=='__main__':unittest.main()
