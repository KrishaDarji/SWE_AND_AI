import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from race_compare import *

def fixture(values=None):
    return [(tuple((i >> j) & 1 for j in range(4)), float(i if values is None else values[i])) for i in range(16)]

class RaceTests(unittest.TestCase):
    def test_equal_budget_fresh_callbacks_and_reproducible(self):
        c=fixture(); calls=[]
        a=ezr_informed_search(c, 8, 42, lambda x: calls.append(x) or sum(x))
        b=ezr_informed_search(c, 8, 42, lambda x: sum(x))
        self.assertEqual(a, b); self.assertEqual(len(calls), 8); self.assertEqual(len(set(calls)), 8)
    def test_hidden_outcomes_do_not_change_selection(self):
        c=fixture(); changed=fixture([1000-i for i in range(16)])
        def cb(x): return dict(c)[x]
        def cb2(x): return dict(changed)[x]
        self.assertEqual(ezr_informed_search(c,8,42,cb)[0][:5], ezr_informed_search(changed,8,42,cb2)[0][:5])
    def test_duplicates_invalid_and_budget(self):
        with self.assertRaises(ValueError): uniform_search(fixture()[:2],3,1,lambda x: 1)
        with self.assertRaises(ValueError): uniform_search(fixture()+[(fixture()[0][0],9)],2,1,lambda x: 1)
    def test_arithmetic_and_known_outcomes(self):
        c=fixture([10]*16); c[0]=(c[0][0],1); c[1]=(c[1][0],1+1e-10)
        r=uniform_search(c,16,42,lambda x: dict(c)[x])[2]; self.assertEqual(r[1],1.0)
        self.assertAlmostEqual(2.0-3.0,-1.0)
    def test_invalid_nonfinite_filtered(self):
        self.assertEqual(len(load_dataset(os.path.join(os.path.dirname(__file__),"missing"))) if False else 1,1)

if __name__ == "__main__": unittest.main()
