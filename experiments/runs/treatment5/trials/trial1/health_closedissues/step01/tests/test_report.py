import csv, math, os, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from report import summarize, DIRECTIONS

class ReportTests(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_complete_path_and_counts(self):
        p = self.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n.1,.8,.5,10,squared_error,1,0,3\n.2,.7,.4,10,squared_error,1,0,3\n.1,.8,.5,10,squared_error,1,0,3\n")
        s = summarize(p)
        self.assertEqual(s["record_count"], 3); self.assertAlmostEqual(s["numeric"]["MRE-"]["mean"], .13333333333333333)
        self.assertEqual(s["categorical"]["criterion"], {"squared_error": 3}); self.assertEqual(s["distinct_configurations"], 1)
        self.assertEqual(s["repeated_configuration_rows"], 3); self.assertEqual(s["duplicate_rows"], 1); self.assertEqual(DIRECTIONS["MRE-"], "minimize")
    def test_missing_nonfinite_invalid_and_header(self):
        p = self.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\nNaN,inf,bad,?,,x,0,2\n")
        s = summarize(p); self.assertEqual(s["missing_values"]["MRE-"], 1); self.assertEqual(s["missing_values"]["N_estimators"], 1)
        h = self.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n")
        z = summarize(h); self.assertEqual(z["record_count"], 0); self.assertEqual(z["numeric"]["MRE-"]["count"], 0)
    def test_roles_and_whitespace(self):
        p = self.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n .1 ,.2,.3, 4 , absolute_error ,2,.5, 6\n")
        s = summarize(p); self.assertEqual(s["roles"]["configuration"][0], "N_estimators"); self.assertEqual(s["categorical"]["criterion"], {"absolute_error": 1})
if __name__ == "__main__": unittest.main()
