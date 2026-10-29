import csv, math, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import certify

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class CertifyTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False); self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name
    def rows(self, ref, cand):
        base = {"ACC+":".5", "PRED40+":".4", "N_estimators":"1", "Min_sample_leaves":"1", "Min_impurity_decrease":"0", "Max_depth":"1"}
        return [{**base, "criterion":"squared_error", "MRE-":str(x)} for x in ref] + [{**base, "criterion":"absolute_error", "MRE-":str(x)} for x in cand]
    def test_direction_and_welch_interval(self):
        result = certify(self.write(self.rows([4, 6, 8], [1, 2, 3])))
        self.assertEqual(result["primary"]["difference_candidate_minus_reference"], -4.0)
        self.assertAlmostEqual(result["primary"]["welch_satterthwaite_df"], 2.9411764705882346)
        self.assertLess(result["primary"]["confidence_interval"][0], result["primary"]["confidence_interval"][1])
    def test_threshold_uncertainty(self):
        result = certify(self.write(self.rows([4, 6, 8], [1, 2, 3])), 2)
        self.assertTrue(result["practical_effect"]["observed_difference_exceeds_threshold"])
        self.assertFalse(result["practical_effect"]["entire_confidence_interval_favorably_beyond_threshold"])
    def test_missing_and_zero_variance(self):
        result = certify(self.write(self.rows([1, 1, float("nan")], [1, 1, "bad"])))
        self.assertEqual(result["primary"]["p_value"], 1.0)
    def test_insufficient_and_invalid_threshold(self):
        with self.assertRaises(ValueError): certify(self.write(self.rows([1], [2, 3])))
        with self.assertRaises(ValueError): certify(self.write(self.rows([1, 2], [3, 4])), -1)

if __name__ == "__main__": unittest.main()
