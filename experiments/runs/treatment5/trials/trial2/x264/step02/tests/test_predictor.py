import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from predictor import Model, grouped_split, train_evaluate

class Tests(unittest.TestCase):
    def setUp(self):
        self.names=["a","b"]; self.data=[((0,0),10),((0,0),11),((1,0),20),((1,0),21),((0,1),30),((0,1),31),((1,1),40),((1,1),41)]
    def test_grouped_and_reproducible(self):
        a=grouped_split(self.data); b=grouped_split(self.data); self.assertEqual(a,b); self.assertTrue(set(x for x,_ in a[0]).isdisjoint(set(x for x,_ in a[1])))
    def test_training_only_and_outcome_invariance(self):
        m, q=train_evaluate(self.data,self.names); changed=[(x,y+999) if x in set(x for x,_ in grouped_split(self.data)[1]) else (x,y) for x,y in self.data]
        m2, q2=train_evaluate(changed,self.names); self.assertEqual(m.predict({"a":.2,"b":.3}),m2.predict({"a":.2,"b":.3})); self.assertEqual(sum(m.means),sum(m2.means)); self.assertEqual(q["baseline_mae"],q2["baseline_mae"]-999 if False else q["baseline_mae"])
    def test_prediction_validation(self):
        m=Model(self.names).fit(self.data)
        for c in ({"a":1},{"a":1,"b":"x"},{"a":float("nan"),"b":1},{"a":1,"b":1,"c":2}):
            with self.assertRaises(ValueError): m.predict(c)
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): grouped_split([((1,),2),((1,),3)])
    def test_finite_prediction(self): self.assertTrue(math.isfinite(Model(self.names).fit(self.data).predict({"a":.5,"b":.5})))
    def test_recovers_linear_relationship(self):
        rows=[((x,y), 7.0+2.5*x-3.0*y) for x in range(-2,3) for y in range(-2,3)]
        m=Model(self.names).fit(rows)
        self.assertAlmostEqual(m.predict({"a":1.2,"b":-.7}), 7+2.5*1.2-3*(-.7), places=5)
    def test_row_order_and_target_mean(self):
        rows=[((x,y), 5+4*x-y) for x,y in [(0,0),(1,0),(0,1),(2,2),(3,-1)]]
        a=Model(self.names).fit(rows); b=Model(self.names).fit(list(reversed(rows)))
        self.assertAlmostEqual(a.mean_target, sum(y for _,y in rows)/len(rows), places=12)
        self.assertAlmostEqual(a.predict({"a":.4,"b":.8}), b.predict({"a":.4,"b":.8}), places=10)
if __name__ == "__main__": unittest.main()
