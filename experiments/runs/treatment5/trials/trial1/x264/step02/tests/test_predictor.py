import math, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from predictor import *

NAMES = ("a", "b")
def rows(): return [((0,0),10),((0,0),11),((0,1),20),((1,0),30),((1,1),40)]

class Tests(unittest.TestCase):
    def test_grouped_reproducible_and_repeated(self):
        a,b=grouped_split(rows()); c,d=grouped_split(rows()); self.assertEqual(a,c); self.assertEqual(b,d); self.assertEqual(set(x for x,_ in a)&set(x for x,_ in b),set())
    def test_training_only_fit_and_outcome_invariance(self):
        tr,te=grouped_split(rows()); m=Predictor(NAMES,tr); m2=Predictor(NAMES,[(x,y+999 if x in set(x for x,_ in te) else y) for x,y in tr])
        self.assertEqual(m.predict({"a":0,"b":1}),m2.predict({"a":0,"b":1})); self.assertEqual(m.target_mean,m2.target_mean)
        self.assertEqual(Predictor(NAMES,tr).means,Predictor(NAMES,tr).means)
    def test_prediction_and_schema_validation(self):
        m=Predictor(NAMES,rows()); self.assertTrue(math.isfinite(m.predict({"a":1,"b":1})))
        for bad in ({"a":1},{"a":1,"b":1,"c":2},{"a":"x","b":1},{"a":float("nan"),"b":1},{"a":float("inf"),"b":1}):
            with self.assertRaises(DataError): m.predict(bad)
    def test_invalid_data_and_insufficient_groups(self):
        with self.assertRaises(DataError): grouped_split([((1,1),2)])
        fd,path=tempfile.mkstemp(text=True); os.close(fd)
        try:
            with open(path,"w") as f: f.write("a,b,Performance-\n1,no,2\n")
            with self.assertRaises(DataError): read_csv(path)
        finally: os.unlink(path)

if __name__ == "__main__": unittest.main(verbosity=2)
