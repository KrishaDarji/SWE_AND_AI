import os, sys, tempfile, unittest
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE,"../source"))
from justify import justify
H="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def fixture(body):
 f=tempfile.NamedTemporaryFile(mode="w",delete=False); f.write(H+body); f.close(); return f.name
def row(m,a,p,n=1,c="x",l=1,d=0,z=1): return f"{m},{a},{p},{n},{c},{l},{d},{z}\n"
class JustifyTests(unittest.TestCase):
 def test_order_and_full_configuration(self):
  p=fixture(row(1,9,9)+row(1,8,9,2,"y",2,1,2)+row(0,1,1,3,"z",3,2,3)+row(0,2,1,4,"q",4,3,4)); r=justify(p); self.assertEqual(r["recommendation"]["position"],5); self.assertEqual(r["recommendation"]["configuration"]["criterion"],"q"); os.unlink(p)
 def test_finite_eligibility_positions_and_even_medians(self):
  p=fixture("bad\n"+row(1,2,3)+row(3,4,5)+"inf,6,7,2,x,2,1,2\n"); r=justify(p); self.assertEqual(r["recommendation"]["position"],3); self.assertEqual(r["references"]["MRE-"]["median"],2); self.assertEqual(r["references"]["ACC+"]["count"],3); os.unlink(p)
 def test_missing_and_empty(self):
  p=fixture("1,2,3,1,,1,0,1\n"); self.assertEqual(justify(p)["status"],"no_eligible_recommendation"); os.unlink(p)
  p=fixture(""); self.assertEqual(justify(p)["status"],"empty"); os.unlink(p)
 def test_each_reference_population_and_association_label(self):
  p=fixture(row(1,2,3)+row(3,2,9,2,"y",2,1,2)+row(5,8,7,3,"z",3,2,3)); r=justify(p); self.assertEqual(r["references"]["MRE-"]["count"],3); self.assertEqual(r["references"]["PRED40+"]["median"],7); self.assertIn("global marginal",r["associations"]["label"]); os.unlink(p)
if __name__ == "__main__": unittest.main()
