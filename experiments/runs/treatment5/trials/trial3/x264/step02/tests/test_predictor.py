import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from predictor import InputError, RidgePredictor, grouped_split

def rows():
    return [((0,0), 10), ((0,0), 11), ((1,0), 20), ((1,0), 21), ((0,1), 30), ((0,1), 31), ((1,1), 40), ((1,1), 41)]

class Tests(unittest.TestCase):
    def test_grouped_reproducible_and_nonempty(self):
        a=grouped_split(rows(),42); b=grouped_split(rows(),42)
        self.assertEqual(a,b); self.assertTrue(set(x for x,_ in a[0]).isdisjoint(set(x for x,_ in a[1])))
    def test_training_only_preprocessing(self):
        m=RidgePredictor(["a","b"]).fit(rows()[:4]); p=m.predict_one({"a":1,"b":0})
        m2=RidgePredictor(["a","b"]).fit(rows()[:4]+[((100,100),9999)])
        self.assertNotEqual(p,m2.predict_one({"a":1,"b":0}))
    def test_prediction_and_validation(self):
        m=RidgePredictor(["a","b"]).fit(rows())
        self.assertTrue(math.isfinite(m.predict_one({"a":1,"b":1})))
        for c in ({"a":1}, {"a":1,"b":1,"c":2}, {"a":"x","b":1}, {"a":float("nan"),"b":1}):
            with self.assertRaises(InputError): m.predict_one(c)
    def test_insufficient_groups_and_outcome_invariance(self):
        with self.assertRaises(InputError): grouped_split([((1,),1),((1,),2)],42)
        m=RidgePredictor(["a"]).fit([((0,),1),((1,),3)])
        self.assertEqual(m.target_mean,2)
        m2=RidgePredictor(["a"]).fit([((0,),100),((1,),3)])
        self.assertNotEqual(m.target_mean,m2.target_mean)

if __name__ == "__main__": unittest.main(verbosity=2)
