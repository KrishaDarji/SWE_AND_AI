import math, sys, unittest
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'source'))
import certify

class TestCertify(unittest.TestCase):
 def test_direction_and_threshold_uncertain(self):
  r=certify.compare([0,1,2,3,4,5],[2,3,4,5,6,7],5); self.assertLess(r['difference_a_minus_b'],0); self.assertTrue(r['point_exceeds_threshold'] is False)
  r=certify.compare([0,20],[0,4],5); self.assertTrue(r['point_exceeds_threshold']); self.assertFalse(r['interval_beyond_threshold'])
 def test_welch_arithmetic(self):
  r=certify.compare([1,2,3],[10,12,14]); self.assertAlmostEqual(r['difference_a_minus_b'],-10,12); self.assertAlmostEqual(r['df'],2.9411764706,8)
 def test_missing_insufficient_zero_invalid(self):
  self.assertRaises(ValueError,certify.summarize,[]); self.assertRaises(ValueError,certify.compare,[1],[2,3]); self.assertRaises(ValueError,certify.compare,[1,1],[2,2]); self.assertRaises(ValueError,certify.compare,[1,2],[3,4],-1)
 def test_reference_numerics(self):
  self.assertAlmostEqual(certify.t_quantile(.975,10),2.22813885196,7); self.assertAlmostEqual(certify.t_cdf(0,10),.5,15); self.assertAlmostEqual(certify.t_cdf(10,10),.999999205,6); self.assertAlmostEqual(certify.t_cdf(1,1),.75,2)
 def test_nonfinite_ignored_by_fixture_logic(self):
  vals=[1,float('nan'),2,float('inf')]; clean=[x for x in vals if math.isfinite(x)]; self.assertEqual(certify.summarize(clean)[0],2)
if __name__=='__main__': unittest.main()
