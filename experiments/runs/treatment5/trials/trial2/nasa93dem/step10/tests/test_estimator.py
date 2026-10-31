import math, os, sys, unittest
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,os.path.join(ROOT,"source"))
import estimator as e

class T(unittest.TestCase):
 def row(self,v,y): return {"x":{k:v for k in e.PREDICTORS},"y":y,"csv_position":9,"idX":"same"}
 def test_mae_and_cases(self):
  self.assertAlmostEqual(e.mae([1,3],[2,1]),1.5); self.assertEqual(e.classify(-.1),"win"); self.assertEqual(e.classify(.1),"loss"); self.assertEqual(e.classify(0),"tie")
 def test_groups_reproducible_and_separated(self):
  rs=[self.row(str(i),i+1) for i in range(8)]; a,b=e.split_groups(rs,42); c,d=e.split_groups(rs,42); self.assertEqual((a,b),(c,d)); self.assertTrue(set(a).isdisjoint(d))
 def test_mean_and_fresh_fit(self):
  rs=[self.row("a",2),self.row("b",4),self.row("c",6)]; m=e.Estimator().fit(rs); self.assertEqual(m.predict(rs[0]),4); self.assertEqual(statistics_mean([r["y"] for r in rs]),4)
 def test_invalid_target_and_exclusions(self):
  self.assertRaises(ValueError,e.Estimator().fit,[self.row("a",math.nan)]); self.assertEqual(len(e.PREDICTORS),22); self.assertNotIn("Effort-",e.PREDICTORS); self.assertEqual(self.row("a",1)["idX"],"same")
 def test_outcome_change_does_not_change_prediction(self):
  a=[self.row("a",2),self.row("b",4),self.row("c",8)]; b=[dict(x=r["x"],y=r["y"],csv_position=9,idX="same") for r in a]; b[2]["y"]=800
  self.assertEqual(e.Estimator().fit(a[:2]).predict(a[0]),e.Estimator().fit(b[:2]).predict(b[0]))
def statistics_mean(x): return sum(x)/len(x)
if __name__=="__main__": unittest.main()
