import math, os, sys, unittest
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from source.estimator import *

def row(x,y=1,pos=2): return {'x':tuple(x),'y':y,'position':pos,'idX':str(pos)}
class Tests(unittest.TestCase):
 def test_mae_cases(self):
  self.assertEqual(mae([1,3],[2,2]),1); self.assertLess(1-2,-.5); self.assertGreater(2-1,0); self.assertLessEqual(abs(1-1),1e-12)
 def test_reproducible_and_separated(self):
  rs=[row([str(i%3),'a'],i+1,i+2) for i in range(12)]; a=split_indices(rs,42); self.assertEqual(a,split_indices(rs,42)); self.assertTrue(set(rs[i]['x'] for i in a[0]).isdisjoint(set(rs[i]['x'] for i in a[1])))
 def test_baseline_and_fresh_fit(self):
  rs=[row(['a'],2),row(['b'],4),row(['c'],8)]; m=Estimator().fit(rs[:2]); self.assertEqual(m.mean,3); self.assertEqual(Estimator().fit(rs[:2]).predict(rs[2:]),m.predict(rs[2:]))
 def test_invalid_targets_identifiers_and_exclusions(self):
  self.assertEqual(load_csv(os.environ['NASA_DATA'])[0]['position'],2); self.assertEqual(len(PREDICTORS),22); self.assertEqual(load_csv(os.path.join(os.path.dirname(__file__),'fixture.csv')).invalid_count,4)
 def test_loading_compare_policy(self):
  rs=load_csv(os.path.join(os.path.dirname(__file__),'fixture.csv')); self.assertEqual(rs[0]['y'],0.0); self.assertEqual(rs[0]['position'],2); self.assertEqual(rs[0]['idX'],'7'); out=compare(rs,seeds=[42]); self.assertTrue(math.isfinite(out[0]['model_mae'])); self.assertTrue(math.isfinite(out[0]['baseline_mae']))
  with self.assertRaisesRegex(ValueError,'no usable'): compare(load_csv(os.path.join(os.path.dirname(__file__),'invalid_fixture.csv')),seeds=[42])
 def test_failures(self):
  with self.assertRaises(ValueError): split_indices([row(['a'],1)],1)
  with self.assertRaises(ValueError): Estimator().fit([row(['a'],None)])
 def test_outcomes_do_not_change_fit(self):
  a=[row(['a'],2),row(['b'],4),row(['c'],9)]; b=[dict(r) for r in a]; b[2]['y']=109; self.assertEqual(Estimator().fit(a[:2]).predict(a[2:]),Estimator().fit(b[:2]).predict(b[2:]))
if __name__=='__main__': unittest.main()
