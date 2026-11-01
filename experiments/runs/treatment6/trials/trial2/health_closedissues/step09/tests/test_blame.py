import csv, os, sys, tempfile, unittest
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,"../source"))
from blame import rank_associations

H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def fixture(body):
 f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.write(H+body); f.close(); return f.name
class BlameTests(unittest.TestCase):
 def test_known_eta_direction_and_only_fields(self):
  p=fixture("1,0,0,1,a,1,0,1\n2,0,0,2,a,2,0,2\n5,0,0,3,b,3,1,3\n6,0,0,4,b,4,1,4\n"); r=rank_associations(p); self.assertEqual({x["field"] for x in r["fields"]},{"N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"}); self.assertAlmostEqual(next(x for x in r["fields"] if x["field"]=="criterion")["score"], 16/17); self.assertGreater(r["ranking"][0]["score"],0); os.unlink(p)
 def test_boundaries_missing_and_per_field_population(self):
  p=fixture("1,0,0,1,a,1,0,1\n2,0,0,2,a,2,0,2\n3,0,0,NA,b,3,1,3\ninf,0,0,4,b,4,1,4\n"); r=rank_associations(p); n=next(x for x in r["fields"] if x["field"]=="N_estimators"); self.assertEqual(n["eligible"],2); self.assertEqual(n["excluded"],2); self.assertTrue(n["boundaries"]); self.assertEqual(next(x for x in r["fields"] if x["field"]=="criterion")["eligible"],3); os.unlink(p)
 def test_constant_empty_insufficient_and_tie(self):
  p=fixture("1,0,0,1,a,1,0,1\n1,0,0,1,a,1,0,1\n"); r=rank_associations(p); self.assertTrue(all(not x["defined"] for x in r["fields"])); os.unlink(p)
  p=fixture(""); r=rank_associations(p); self.assertTrue(all(x["eligible"]==0 for x in r["fields"])); self.assertEqual(r["ranking"],[]); os.unlink(p)
if __name__ == "__main__": unittest.main()
