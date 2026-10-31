import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from predictor import ConfigPredictor

def rows():
    out=[]
    for i in range(12):
        out.append({"MRE-": str(i/20), "ACC+": str(1-i/30), "PRED40+":"99", "N_estimators":str(10+i%4), "criterion":["a","b","c"][i%3], "Min_sample_leaves":str(1+i%3), "Min_impurity_decrease":str(i%5), "Max_depth":str(2+i%4)})
    return out

class T(unittest.TestCase):
    def test_groups_repro_and_exclusion(self):
        a,b=ConfigPredictor(rows()),ConfigPredictor(rows()); self.assertEqual(a.test_keys,b.test_keys); self.assertEqual(a.evaluate(),b.evaluate())
        self.assertNotIn("PRED40+", __import__('predictor').PREDICTORS)
    def test_prediction_and_schema(self):
        m=ConfigPredictor(rows()); p=m.predict({k: rows()[0][k] for k in __import__('predictor').PREDICTORS}); self.assertEqual(set(p),{"MRE-","ACC+"}); self.assertTrue(all(math.isfinite(x) for x in p.values()))
        with self.assertRaises(ValueError): m.predict({"criterion":"a"})
        with self.assertRaises(ValueError): m.predict({x:"1" for x in ["N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth","extra"]})
    def test_missing_unseen_and_zero(self):
        m=ConfigPredictor(rows()); c={"N_estimators":"bad","criterion":"new","Min_sample_leaves":"", "Min_impurity_decrease":"nan","Max_depth":"4"}; self.assertTrue(math.isfinite(m.predict(c)["MRE-"]))
        z=rows(); z[0]["MRE-"]=z[0]["ACC+"]= "0"; self.assertIsNotNone(ConfigPredictor(z))
    def test_outcome_changes_do_not_fit(self):
        a=ConfigPredictor(rows()); r=rows(); [r[i].update({"MRE-":"100","ACC+":"-100"}) for i in range(len(r)) if __import__('predictor')._key(r[i]) in a.test_keys]; b=ConfigPredictor(r)
        c={k: rows()[1][k] for k in __import__('predictor').PREDICTORS}; self.assertEqual(a.predict(c), b.predict(c)); self.assertEqual(a.mean,b.mean)
    def test_invalid_and_insufficient(self):
        r=rows(); r[0]["MRE-"]="no"; 
        self.assertGreater(ConfigPredictor(r).evaluate()["train_rows"], 0)
        one=[dict(rows()[0]) for _ in range(4)]
        with self.assertRaises(ValueError): ConfigPredictor(one)

if __name__ == "__main__": unittest.main(verbosity=2)
