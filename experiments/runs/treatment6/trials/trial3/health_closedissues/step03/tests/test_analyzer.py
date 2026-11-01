import csv, os, tempfile, unittest
from source.analyzer import analyze

def fixture(text):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    f.write(text); f.close(); return f.name

class AnalyzerTests(unittest.TestCase):
    def test_roles_frequencies_medians_dispersion(self):
        p=fixture("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n1,2,3,1, 1 ,2,0.1,3\n2,4,6,1,1,4,0.2,5\n3,6,9,2,2,6,0.3,7\n")
        x=analyze(p); self.assertEqual(x["roles"]["criterion"],"configuration (categorical)")
        self.assertEqual(x["criterion_frequencies"],{"1":2,"2":1}); self.assertEqual(x["outcomes"]["MRE-"]["median"],2)
        self.assertAlmostEqual(x["outcomes"]["MRE-"]["sample_sd"],1); self.assertEqual(x["outcomes"]["MRE-"]["iqr"],1)
    def test_invalid_missing_all_missing_and_positions(self):
        p=fixture("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n?,nan,2,bad, 2 ,?,,4\n1,2,3,1,1,2,0.1,3\n1,2\n\n")
        x=analyze(p); self.assertEqual(x["malformed_row_positions"],[4]); self.assertEqual(x["invalid_nonfinite_numeric"]["N_estimators"],1)
        self.assertEqual(x["normalized_configurations"][0]["row_positions"],[2]); self.assertEqual(x["normalized_configurations"][1]["row_positions"],[3])
    def test_repeats_vs_exact_duplicates_numeric_equivalence(self):
        h="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
        p=fixture(h+"1,2,3,1.0, squared_error ,2,0.10,3\n1,2,3,1, squared_error,2,0.1,3\n1,2,3,1, squared_error,2,0.1,3\n")
        x=analyze(p); self.assertEqual(len(x["repeated_configurations"]),1); self.assertEqual(x["repeated_configurations"][0]["row_positions"],[2,3,4]); self.assertEqual(x["exact_duplicate_row_groups"],[[3,4]])
    def test_empty_and_header_only(self):
        self.assertEqual(analyze(fixture(""))["record_count"],0)
        self.assertEqual(analyze(fixture("a,b\n"))["record_count"],0)
if __name__ == "__main__": unittest.main()
