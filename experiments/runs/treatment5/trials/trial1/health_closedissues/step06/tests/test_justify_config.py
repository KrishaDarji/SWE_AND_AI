import csv, math, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import justify_config as j

HEADER = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
def fixture(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    f.write(",".join(HEADER)+"\n"+text); f.close(); return f.name

class Tests(unittest.TestCase):
    def test_selection_and_tiebreakers(self):
        p=fixture(".1,.5,.2,1,c,1,0,1\n.1,.6,.1,2,c,2,0,2\n.1,.6,.3,3,c,3,0,3\n")
        a=j.analyze(p); self.assertEqual(a["chosen"]["record_id"],"3")
    def test_medians_evidence_and_identifier(self):
        p=fixture(".2,.4,.6,1,c,1,0,1\n.4,.8,.2,2,c,2,0,2\n")
        a=j.analyze(p); self.assertAlmostEqual(a["references"]["MRE-"][0],.3); self.assertEqual(a["references"]["MRE-"][1],2); out=j.render(p)
        self.assertIn("observed record 1",out); self.assertIn("Measured outcomes",out); self.assertIn("reference median",out)
    def test_missing_nonfinite_excluded(self):
        p=fixture(".2,.4,?,1,c,1,0,1\nNaN,.8,.2,2,c,2,0,2\n.4,.8,.2,3,c,3,0,3\n")
        a=j.analyze(p); self.assertEqual(a["eligible_count"],1); self.assertEqual(a["chosen"]["record_id"],"3")
    def test_no_eligible(self):
        p=fixture("?,.4,.2,1,c,1,0,1\ninf,.8,.2,2,c,2,0,2\n")
        self.assertIsNone(j.analyze(p)["chosen"]); self.assertIn("no eligible",j.render(p).lower())
    def test_all_outcome_references(self):
        p=fixture(".2,.4,.6,1,c,1,0,1\n.4,.8,.2,2,c,2,0,2\n")
        a=j.analyze(p); self.assertAlmostEqual(a["references"]["ACC+"][0],.6); self.assertEqual(a["references"]["ACC+"][1],2); self.assertAlmostEqual(a["references"]["PRED40+"][0],.4); self.assertEqual(a["references"]["PRED40+"][1],2)
    def test_references_include_ineligible_rows(self):
        p=fixture(".2,.4,?,1,c,1,0,1\n.4,?,.2,2,c,2,0,2\n?,.8,.6,3,c,3,0,3\n")
        a=j.analyze(p)
        self.assertEqual(a["eligible_count"],0)
        self.assertIsNone(a["chosen"])
        self.assertAlmostEqual(a["references"]["MRE-"][0],.3); self.assertEqual(a["references"]["MRE-"][1],2)
        self.assertAlmostEqual(a["references"]["ACC+"][0],.6); self.assertEqual(a["references"]["ACC+"][1],2)
        self.assertAlmostEqual(a["references"]["PRED40+"][0],.4); self.assertEqual(a["references"]["PRED40+"][1],2)

if __name__ == "__main__": unittest.main()
