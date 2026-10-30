import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'source'))
from certify import *

class TestCertify(unittest.TestCase):
    def test_direction_and_reference_values(self):
        r=welch([1,2,3],[4,5,6], 1)
        self.assertLess(r['difference'],0); self.assertAlmostEqual(r['df'],4.0,12); self.assertAlmostEqual(r['t'],-3.674234614,7)
        self.assertAlmostEqual(t_cdf(1.959963984540, 1e7), .975, 5)
        self.assertAlmostEqual(t_cdf(2.228138851,10), .975, 6)
    def test_missing_and_insufficient(self):
        self.assertEqual(summarize([1,float('nan'),2])[0],2)
        with self.assertRaises(ValueError): welch([1],[2,3])
    def test_zero_and_threshold(self):
        with self.assertRaises(ValueError): welch([1,1],[1,1])
        with self.assertRaises(ValueError): welch([1,2],[3,4],-1)
        r=welch([0,10,20,30],[0,1,2,3], 5)
        self.assertTrue(r['point_exceeds_threshold']); self.assertTrue(r['practical_uncertain'])
    def test_large_statistic(self):
        r=welch([100,100.1,99.9],[0,0.1,-0.1], 5)
        self.assertGreater(r['t'],100)

if __name__ == '__main__': unittest.main()
