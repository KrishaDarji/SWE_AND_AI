import math,os,sys,unittest
sys.path.insert(0,os.path.join(os.path.dirname(__file__),"..","source"))
from tuner import race,random_search,compare,summarize
def fixture(n=40): return [{"id":i+1,"config":(i,i%3)} for i in range(n)]
class Tests(unittest.TestCase):
 def test_callback_observation_and_privacy(self):
  cs=fixture(); seen=[]
  def cb(c): self.assertEqual(set(c),{"id","config"}); seen.append(c["id"]); return 1000-c["id"]*7
  r=race(cs,20,42,cb); self.assertEqual(r["measured"],1000-r["recommended"]["id"]*7); self.assertEqual(len(seen),20)
 def test_equal_budget_reproducibility_and_hidden_outcomes(self):
  cs=fixture(); cb=lambda c:c["id"]*13+0.25
  self.assertEqual(race(cs,20,42,cb),race(cs,20,42,cb)); self.assertEqual(race(cs,20,42,cb)["order"],race(cs,20,42,lambda c:999)["order"])
  self.assertEqual(random_search(cs,20,42,cb)["order"],random_search(cs,20,42,lambda c:-7)["order"])
 def test_duplicate_configs_and_both_strategies(self):
  cs=[{"id":1,"config":(1,0)},{"id":2,"config":(1,0)},{"id":3,"config":(0,1)},{"id":4,"config":(0,0)}]
  for fn in (race,random_search):
   calls=[]; r=fn(cs,3,2,lambda c:(calls.append(c["id"]) or c["id"])); self.assertEqual(len(calls),3); self.assertEqual(len(set(calls)),3); self.assertEqual(r["order"],calls)
 def test_summary_known_behavior(self):
  s=summarize([-10,0,10],0); self.assertEqual((s["wins"],s["ties"],s["losses"]),(1,1,1)); self.assertEqual(s["mean"],0); self.assertEqual(s["median"],0)
  s=summarize([-1,-.1,.1],.2); self.assertEqual((s["wins"],s["ties"],s["losses"]),(1,2,0))
 def test_invalid_budgets_and_observations(self):
  cs=fixture()
  for b in (True,False,1.5,0,41):
   with self.assertRaises(ValueError): race(cs,b,1,lambda c:1)
  for v in (math.nan,math.inf,True,"x"):
   with self.assertRaises(ValueError): random_search(cs,2,1,lambda c,v=v:v)
 def test_compare_shape(self):
  out,summary=compare(os.environ["X264_DATA"],20,range(42,52)); self.assertEqual(len(out),20); self.assertEqual(summary["count"],10); self.assertTrue(all(x["budget"]==20 for x in out))
if __name__=="__main__": unittest.main(verbosity=2)
