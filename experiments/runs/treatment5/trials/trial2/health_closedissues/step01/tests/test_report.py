import csv, json, os, tempfile, unittest, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from report import report

H = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
class TestReport(unittest.TestCase):
    def make(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_complete_path(self):
        p = self.make(H + "1,2,3,10, squared_error ,1,0.1,4\n1,2,3,10,squared_error,1,0.1,4\n?,nan,,20,absolute_error,2,0.2,5\n")
        x = report(p)
        self.assertEqual(x["record_count"], 3); self.assertEqual(x["categorical_frequencies"]["criterion"]["squared_error"], 2)
        self.assertEqual(x["distinct_configurations"], 2); self.assertEqual(x["duplicate_rows"], 1)
        self.assertEqual(x["roles"]["MRE-"], "outcome"); self.assertEqual(x["optimization_directions"]["MRE-"], "minimize")
        self.assertEqual(x["missing_or_invalid"]["MRE-"], 1); self.assertEqual(x["missing_or_invalid"]["N_estimators"], 0)
    def test_empty_and_header_only(self):
        p = self.make(""); self.assertEqual(report(p)["record_count"], 0)
        p = self.make(H); x = report(p); self.assertEqual(x["record_count"], 0); self.assertEqual(x["distinct_configurations"], 0)
    def test_nonfinite_and_roles(self):
        p = self.make(H + "inf,1,2,1,c,1,0,1\n"); x = report(p)
        self.assertEqual(x["missing_or_invalid"]["MRE-"], 1); self.assertEqual(x["roles"]["criterion"], "configuration")
if __name__ == "__main__": unittest.main()
