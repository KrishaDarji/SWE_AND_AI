import csv, os, tempfile, unittest
from source.analyzer import analyze, blame

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
    def test_blame_score_direction_and_eligible_population(self):
        h="MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
        p=fixture(h+"0,1,1,1,a,1,0.1,1\n2,1,1,1,a,1,0.1,2\n10,1,1,2,b,2,0.2,3\n12,1,1,2,b,2,0.2,4\n?,1,1,3,b,3,0.3,5\n")
        x=blame(p); self.assertEqual(set(z["field"] for z in x["ranking"]), set(x["fields"]))
        first=next(z for z in x["ranking"] if z["field"]=="N_estimators")
        self.assertEqual(first["eligible_count"],4); self.assertAlmostEqual(first["reference_mean"],6)
        self.assertAlmostEqual(first["eta_squared"], 100/104)
        self.assertEqual(first["groups"][0]["mean_mre"],1)
    def test_blame_constant_empty_and_tie_safe(self):
        h="MRE-,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
        x=blame(fixture(h+"1,1,a,1,0,1\n1,1,a,1,0,1\n"))
        self.assertTrue(all(z["eta_squared"] is None for z in x["ranking"]))
        self.assertEqual([z["field"] for z in blame(fixture(""))["ranking"]], list(blame(fixture(""))["fields"]))
if __name__ == "__main__": unittest.main()
