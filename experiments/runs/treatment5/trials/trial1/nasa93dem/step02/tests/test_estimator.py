import os, sys, unittest, tempfile, csv
HERE=os.path.dirname(__file__); SRC=os.path.abspath(os.path.join(HERE,"..","source")); sys.path.insert(0,SRC)
from estimator import *

class EstimatorTests(unittest.TestCase):
    def setUp(self): self.names=["a","b"]; self.rows=[[" x ","h"],["x","h"],["y","l"],["z","?"]]; self.y=[1,3,10,20]
    def test_clean_and_schema(self):
        m=CategoricalEstimator(self.names).fit(self.rows,self.y); self.assertIn(MISSING,m._vector({"a":"z","b":"?"})); self.assertEqual(set(m.names),{"a","b"})
    def test_prediction_and_unseen(self):
        m=CategoricalEstimator(self.names).fit(self.rows,self.y); self.assertEqual(m.predict_one({"a":"x","b":"h"}),2); self.assertTrue(math.isfinite(m.predict_one({"a":"new","b":"new"})))
    def test_proposal_validation(self):
        m=CategoricalEstimator(self.names).fit(self.rows,self.y)
        with self.assertRaises(ValueError): m.predict_one({"a":"x"})
        with self.assertRaises(ValueError): m.predict_one({"a":"x","b":"h","c":"q"})
    def test_group_split_reproducible_and_never_all(self):
        a=grouped_split(self.rows,[0,1]); self.assertEqual(a,grouped_split(self.rows,[0,1])); self.assertTrue(set(a[0]).isdisjoint(a[1]))
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): grouped_split([["x"],["x"]],[0])
    def test_actual_missing(self):
        m=CategoricalEstimator(self.names).fit([["x","h"],["x",MISSING]],[1,2]); self.assertEqual(m.predict_one({"a":"x","b":"?"}),2)
    def test_outcome_independence(self):
        m1=CategoricalEstimator(self.names).fit(self.rows[:3],self.y[:3]); m2=CategoricalEstimator(self.names).fit(self.rows[:3], [100,300,1000]); self.assertNotEqual(m1.mean,m2.mean)
    def test_heldout_changes_do_not_enter_fit(self):
        train=self.rows[:2]; m1=CategoricalEstimator(self.names).fit(train,self.y[:2]); m2=CategoricalEstimator(self.names).fit(train,self.y[:2])
        changed=[list(r) for r in self.rows[2:]]; changed[0][0]="new descriptor"; changed[1][1]="new category"
        self.assertEqual(m1.predict_one({"a":"x","b":"h"}),m2.predict_one({"a":"x","b":"h"})); self.assertEqual(m1.mean,m2.mean); self.assertEqual(m1.levels,m2.levels)

    def test_evaluate_excludes_invalid_targets_even_if_held_out(self):
        fd,path=tempfile.mkstemp(dir=HERE,suffix=".csv"); os.close(fd)
        try:
            with open(path,"w",newline="") as f:
                w=csv.writer(f); w.writerow(["a","b","Effort-"])
                for i,target in enumerate(["1","2","nan","inf","oops","5"]): w.writerow(["g"+str(i),"h",target])
            result=evaluate(path); self.assertEqual(result["excluded_targets"],{"missing":0,"invalid":3}); self.assertTrue(math.isfinite(result["mae"])); self.assertTrue(math.isfinite(result["baseline_mae"]))
        finally: os.unlink(path)

    def test_evaluate_no_usable_targets(self):
        fd,path=tempfile.mkstemp(dir=HERE,suffix=".csv"); os.close(fd)
        try:
            with open(path,"w",newline="") as f:
                w=csv.writer(f); w.writerow(["a","b","Effort-"]); w.writerow(["x","h","NaN"])
            with self.assertRaisesRegex(ValueError,"no usable records"): evaluate(path)
        finally: os.unlink(path)

if __name__=="__main__": unittest.main()
