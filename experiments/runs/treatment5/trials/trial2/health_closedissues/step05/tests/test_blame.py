import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import blame

def write(rows):
    f = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False, suffix=".csv")
    w = csv.DictWriter(f, fieldnames=["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"])
    w.writeheader(); w.writerows(rows); f.close(); return f.name

class TestBlame(unittest.TestCase):
    def test_known_mixed_associations_and_exclusions(self):
        rows=[]
        for i in range(12):
            rows.append({"MRE-": str(1 if i < 6 else 10), "ACC+":"999", "PRED40+":"0", "N_estimators":str(i), "criterion":"a" if i<6 else "b", "Min_sample_leaves":"1", "Min_impurity_decrease":str(i), "Max_depth":str(i%2)})
        r=blame.analyze(write(rows)); self.assertIn(r[0]["setting"], {"N_estimators", "criterion"}); self.assertGreater(r[0]["score"], .9); self.assertEqual(r[0]["n"],12)
    def test_missing_constant_all_missing_and_determinism(self):
        rows=[{"MRE-":"1","ACC+":"2","PRED40+":"3","N_estimators":"","criterion":"x","Min_sample_leaves":"4","Min_impurity_decrease":"nan","Max_depth":"2"}, {"MRE-":"2","ACC+":"9","PRED40+":"8","N_estimators":"2","criterion":"x","Min_sample_leaves":"4","Min_impurity_decrease":"","Max_depth":"2"}]
        a=blame.analyze(write(rows)); b=blame.analyze(write(rows)); self.assertEqual([x["setting"] for x in a],[x["setting"] for x in b]); missing=next(x for x in a if x["setting"] == "Min_impurity_decrease"); self.assertEqual(missing["n"],0); self.assertEqual(missing["score"],0.0)
    def test_quantile_boundaries_are_outcome_independent(self):
        gs,b=blame.quantile_groups([1,2,3,4,5,6]); self.assertEqual(len(b),5); self.assertEqual(len(gs),6); self.assertTrue(all(isinstance(x,int) for x in gs))

if __name__ == "__main__": unittest.main()
