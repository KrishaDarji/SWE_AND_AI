import csv, os, tempfile, unittest
from source.analyzer import predict_dataset, analyze, _Tree, predict_configuration

H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
class PredictionTests(unittest.TestCase):
    def make(self, body):
        f=tempfile.NamedTemporaryFile(mode="w",delete=False,newline=""); f.write(H+body); f.close(); self.addCleanup(lambda:os.unlink(f.name)); return f.name
    def test_grouped_reproducible_and_outcomes_excluded(self):
        p=self.make("\n".join(f"{i/10},{i/20},0,{i}, c,1,0,{i%3+1}" for i in range(1,9)))
        a=predict_dataset(p); b=predict_dataset(p); self.assertEqual(a,b); self.assertEqual(a["predictors"], ["N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]); self.assertEqual(a["outcome_exclusions"], ["MRE-","ACC+","PRED40+"])
    def test_zero_invalid_and_schema(self):
        p=self.make("0,0,0,1,c,1,0,1\n1,nan,0,2,c,1,0,2\n2,2,0,3,c,1,0,3\n3,3,0,4,c,1,0,4")
        x=predict_dataset(p); self.assertEqual(x["targets"]["MRE-"]["evaluated_test_targets"],1); self.assertEqual(x["targets"]["MRE-"]["excluded_test_targets"],0)
        with self.assertRaises(ValueError): predict_configuration({},None)
    def test_constant_numeric_and_whitespace_normalize(self):
        p=self.make("1,1,0,1, c ,1,0,1\n1,1,0,1,c,1.0,0.0,1.0")
        x=analyze(p); self.assertEqual(len(x["normalized_configurations"]),1)
if __name__ == "__main__": unittest.main()
