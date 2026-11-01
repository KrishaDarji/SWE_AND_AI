import math, pathlib, sys, unittest
ROOT=pathlib.Path(__file__).parents[1]; sys.path.insert(0,str(ROOT/'source'))
import nasa93_summary as ns

class CertifyTests(unittest.TestCase):
    def rows(self, values):
        return [ns.Project(i+2, {'tool':tool,'Effort-':str(x)}) for i,(tool,x) in enumerate(values)]
    def test_direction_stats_and_inference(self):
        r=ns.certify(self.rows([(' n ',10),('n',12),('h',7),('h',9),('x',100)]), 1)
        self.assertEqual(r['groups']['n']['count'],2); self.assertEqual(r['groups']['h']['sample_variance'],2)
        self.assertEqual(r['difference'],-3); self.assertGreater(r['inference']['df'],1); self.assertLess(r['inference']['ci95'][0],-3)
    def test_threshold_and_validation(self):
        r=ns.certify(self.rows([('n',99),('n',101),('h',83),('h',87)]),1)
        self.assertTrue(r['practical']['point_estimate_meets']); self.assertTrue(r['practical']['entire_ci_supports'])
        for x in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): ns.certify(self.rows([('n',1),('n',2),('h',1),('h',2)]),x)
    def test_missing_absent_insufficient_zero_se(self):
        r=ns.certify(self.rows([('n',1),('n',2),('h',1),('h',2),('h',float('nan'))]))
        self.assertEqual(r['excluded'],1)
        self.assertIn('undefined',ns.certify(self.rows([('n',1),('h',2)]))['inference']['status'])
        self.assertIn('zero standard',ns.certify(self.rows([('n',1),('n',1),('h',2),('h',2)]))['inference']['status'])
    def test_t_reference_values(self):
        self.assertAlmostEqual(ns.t_cdf(0,1),.5,places=12); self.assertAlmostEqual(ns.t_cdf(1.812461,10),.95,places=5)
        self.assertAlmostEqual(ns.t_cdf(1.96,1000000),.975002,places=5)
if __name__=='__main__': unittest.main()
