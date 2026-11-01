import csv, math, os, tempfile, unittest
import sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,"../source"))
from summarize_runs import analyze, grouped_split, evaluate, predict_configuration, CONFIG

H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def fixture(body):
    f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.write(H+body); f.close(); return f.name
class SummaryTests(unittest.TestCase):
 def test_roles_stats_medians_and_categorical(self):
  p=fixture("1,2,3,1, 123 ,2,0.5,4\n2,4,6,1.0,123,2,0.5,4\n3,6,9,2,abc,4,1,8\n"); a=analyze(p)
  self.assertEqual(a["roles"]["criterion"],"configuration"); self.assertEqual(a["criterion"],{"123":2,"abc":1}); self.assertEqual(a["outcomes"]["MRE-"]["median"],2); self.assertEqual(a["outcomes"]["MRE-"]["sample_stddev"],1); self.assertEqual(a["numeric"]["N_estimators"]["median"],1); os.unlink(p)
 def test_invalid_missing_all_missing_and_duplicates(self):
  p=fixture("1,2,3,1,x,2,0.5,4\nNA,inf,3,bad,x,?,nan,4\n1,2,3,1,x,2,0.5,4\n"); a=analyze(p); self.assertEqual(a["invalid"]["N_estimators"],1); self.assertEqual(a["invalid"]["ACC+"],1); self.assertEqual(a["configurations"]["distinct"],1); self.assertEqual(a["configurations"]["repeated"][0]["count"],2); self.assertEqual(a["configurations"]["exact_duplicate_rows"],1); os.unlink(p)
 def test_positions_malformed_empty_header_only(self):
  p=fixture("1,2,3,1,x,2,0.5\n"); a=analyze(p); self.assertEqual(a["malformed"],[2]); os.unlink(p)
  f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.close(); self.assertTrue(analyze(f.name)["empty"]); os.unlink(f.name)
 def test_grouping_prediction_and_schema(self):
  body="".join(f"{i/10},{i/20},0,{i%2+1},c{i%2},2,0.1,4\n" for i in range(10))
  p=fixture(body); a=evaluate(p); self.assertEqual(a["train_groups"]+a["test_groups"],2); self.assertEqual(a["test_rows"],5)
  self.assertEqual(set(predict_configuration(a,{c:("x" if c=="criterion" else 1) for c in CONFIG})),{"MRE-","ACC+"})
  with self.assertRaises(ValueError): predict_configuration(a,{"criterion":"x"})
  os.unlink(p)
 def test_zero_invalid_and_insufficient(self):
  p=fixture("0,0,0,1,x,2,0.1,4\ninf,na,0,2,y,2,0.1,4\n1,1,0,3,y,2,0.1,4\n"); a=evaluate(p); self.assertEqual(a["targets"]["MRE-"]["finite_test"],1); os.unlink(p)
  p=fixture("1,2,3,1,x,2,0.1,4\n");
  with self.assertRaises(ValueError): evaluate(p)
  os.unlink(p)
if __name__ == "__main__": unittest.main()
