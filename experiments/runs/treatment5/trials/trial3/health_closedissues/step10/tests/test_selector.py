import csv, os, tempfile, unittest
from source.selector import *

FIELDS = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
def row(m, a=0, p=0, n=10, c="x", l=1, d=0, h=1):
    return dict(zip(FIELDS, map(str, [m,a,p,n,c,l,d,h])))
class Tests(unittest.TestCase):
    def test_ordering_and_ties(self):
        fs=[(9,row(1,1,1)),(3,row(1,1,1)),(2,row(1,1,0)),(1,row(1,0,1)),(0,row(2,1,1))]
        self.assertEqual(selector(fs)[0], 3)
    def test_eligibility_cap_and_counts_ids(self):
        good=row(1,n=100); bad=row(1,n=101); bad["criterion"]=""; invalid=row(1); invalid["ACC+"]="nan"
        self.assertTrue(eligible(good)); self.assertFalse(eligible(bad)); self.assertFalse(eligible(bad)); self.assertFalse(eligible(invalid))
    def test_rng_reproducible_independent(self):
        fs=[(i,row(i)) for i in range(4)]
        self.assertEqual(baseline(fs,42), baseline(fs,42)); self.assertEqual(baseline(fs,42), random.Random(42).choice(fs))
    def test_summary_arithmetic(self):
        d=[(0,row(0)),(1,row(1))]; self.assertAlmostEqual(d[1][1]["MRE-"] and float(d[1][1]["MRE-"])-float(d[0][1]["MRE-"]),1)
    def test_invalid_and_empty(self):
        with self.assertRaises(ValueError): eligible(row(1), float("inf"))
        with self.assertRaises(ValueError): selector([])
if __name__ == "__main__": unittest.main()
