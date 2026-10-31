import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../source"))
from estimator import *

NAMES = ["a", "b"]
def rows(): return [{"a":" h ","b":"l","Effort-":"1"},{"a":"h","b":"l","Effort-":"3"},{"a":"x","b":"m","Effort-":"5"},{"a":"y","b":"m","Effort-":"7"}]

class Tests(unittest.TestCase):
 def test_exclusions_and_whitespace(self):
  self.assertEqual(predictor_names([{"a":"x","zX":"q","Kloc+":"1","Effort-":"2","Defects-":"3"}]),["a"])
  m=CategoricalEstimator(NAMES).fit(rows()); self.assertEqual(m.predict_one({"a":"h","b":"l"}),2)
 def test_grouped_reproducible(self):
  self.assertEqual(grouped_split(rows()),grouped_split(rows()))
  tr,te=grouped_split(rows()); self.assertTrue(set(tr).isdisjoint(te))
  self.assertNotEqual(tuple(clean(rows()[tr[0]].get(k)) for k in NAMES), tuple(clean(rows()[te[0]].get(k)) for k in NAMES))
 def test_missing_unseen_and_schema(self):
  m=CategoricalEstimator(NAMES).fit(rows()); self.assertEqual(m.predict_one({"a":"","b":"unknown"}),m.mean)
  with self.assertRaises(ValueError): m.predict_one({"a":"h"})
  with self.assertRaises(ValueError): m.predict_one({"a":"h","b":"l","c":"x"})
 def test_outcome_independence(self):
  base=rows(); a=CategoricalEstimator(NAMES).fit(base); changed=rows(); changed[2]["Effort-"]="999999"
  b=CategoricalEstimator(NAMES).fit(changed)
  proposal={k:base[0][k] for k in NAMES}
  self.assertEqual(a.predict_one(proposal),b.predict_one(proposal))
 def test_insufficient_groups(self):
  with self.assertRaises(ValueError): grouped_split([{"a":"x","Effort-":"1"}])
 def test_proposal(self): self.assertTrue(math.isfinite(CategoricalEstimator(NAMES).fit(rows()).predict_one({"a":"new","b":"new"})))

if __name__ == '__main__': unittest.main()
