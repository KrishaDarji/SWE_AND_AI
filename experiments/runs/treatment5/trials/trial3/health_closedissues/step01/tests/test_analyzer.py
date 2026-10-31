import csv, json, pathlib, tempfile, unittest
import sys
HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "source"))
from analyzer import summarize

class AnalyzerTests(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, dir=HERE, encoding="utf-8")
        f.write(text); f.close(); return f.name
    def test_complete_path_and_roles(self):
        p = self.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n1,2,3,10,squared_error,2,0.5,4\n1,2,3,10,squared_error,2,0.5,4\n?,NaN,inf,20,absolute_error,,1.5,6\n")
        x = summarize(p)
        self.assertEqual(x["record_count"], 3); self.assertEqual(x["numeric_statistics"]["MRE-"]["count"], 2)
        self.assertEqual(x["categorical_frequencies"]["criterion"], {"absolute_error": 1, "squared_error": 2})
        self.assertEqual(x["configuration_count_distinct"], 2); self.assertEqual(x["duplicate_rows"], 1)
        self.assertEqual(x["missing_values"]["MRE-"], 1); self.assertEqual(x["invalid_or_nonfinite_numeric"]["ACC+"], 1)
        self.assertEqual(x["outcome_fields"], ["MRE-", "ACC+", "PRED40+"]); self.assertEqual(x["optimization_directions"]["MRE-"], "minimize")
    def test_empty_and_header_only(self):
        p = self.write(""); self.assertTrue(summarize(p)["empty_input"])
        p = self.write("MRE-,criterion\n"); x = summarize(p); self.assertTrue(x["header_only"]); self.assertEqual(x["record_count"], 0)
    def test_whitespace_and_repeated_configuration(self):
        p = self.write(" MRE- , ACC+ , PRED40+ , N_estimators , criterion , Min_sample_leaves , Min_impurity_decrease , Max_depth \n 1 , 2 , 3 , 4 , x , 5 , 6 , 7 \n")
        x = summarize(p); self.assertEqual(x["columns"], ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"])
        self.assertEqual(x["configuration_count_distinct"], 1)

if __name__ == "__main__": unittest.main()
