import csv, os, tempfile, unittest, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import summarize, split_groups, NearestNeighborModel, load, certify, _t_cdf

class SummaryTests(unittest.TestCase):
    def run_case(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            handle.write(text); handle.close(); return summarize(handle.name)
        finally:
            os.unlink(handle.name)
    def test_real_shape_and_stats(self):
        result = summarize(os.environ["X264_DATASET"])
        self.assertGreater(result["record_count"], 0); self.assertEqual(result["outcome"], "Performance-")
        self.assertEqual(result["direction"], "lower is better"); self.assertEqual(result["invalid_outcomes"], 0)
    def test_frequencies_statistics(self):
        result = self.run_case("a,b,Performance-\n1,x,2\n1,y,4\n0,x,6\n")
        self.assertEqual(result["frequencies"]["a"], {"1": 2, "0": 1})
        self.assertEqual(result["outcome_stats"]["count"], 3); self.assertEqual(result["outcome_stats"]["median"], 4)
    def test_missing_invalid(self):
        result = self.run_case("a,Performance-\n?,\n1,bad\n")
        self.assertEqual(result["missing_outcomes"], 1); self.assertEqual(result["invalid_outcomes"], 1)
    def test_empty_header_only_and_no_outcome(self):
        self.assertTrue(summarize(tempfile.NamedTemporaryFile(mode="w", delete=False).name)["missing_input"])
        self.assertEqual(self.run_case("a,b\n")["record_count"], 0)
        self.assertIsNone(self.run_case("a\n1\n")["outcome"])

if __name__ == "__main__": unittest.main()

class PredictionTests(unittest.TestCase):
    def test_prediction_grouping_validation_and_zero(self):
        text = "a,b,Performance-\n0,x,0\n0,x,2\n1,x,10\n1,y,12\n2,y,20\n"
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            handle.write(text); handle.close(); _, rows, _ = load(handle.name)
            train, test = split_groups(rows, ["a", "b"]); self.assertTrue(train and test)
            model = NearestNeighborModel(["a", "b"], "Performance-").fit(train)
            self.assertIsInstance(model.predict_one({"a": "0", "b": "x"}), float)
            with self.assertRaises(ValueError): model.predict_one({"a": "0"})
            self.assertEqual(model.predict_one({"a": "9", "b": "x"}), model.mean)
        finally: os.unlink(handle.name)

    def test_outcome_cannot_be_a_feature_and_zero_is_valid(self):
        model = NearestNeighborModel(["a"], "Performance-").fit([{"a":"x", "Performance-":"0"}, {"a":"y", "Performance-":"2"}])
        self.assertEqual(model.predict_one({"a":"x"}), 0)
        with self.assertRaises(ValueError): model.predict_one({"a":"x", "Performance-":"9", "extra":"z"})

class CertifyTests(unittest.TestCase):
    def case(self, text, threshold=None):
        h=tempfile.NamedTemporaryFile(mode="w", delete=False); h.write(text); h.close()
        try: return certify(h.name, threshold)
        finally: os.unlink(h.name)
    def test_direction_variance_and_inference(self):
        r=self.case("no_8x8dct,Performance-\n0,10\n0,14\n0,18\n1,4\n1,8\n1,12\n")
        self.assertEqual(r["reference"]["variance"],16); self.assertEqual(r["candidate"]["mean"],8); self.assertLess(r["inference"]["difference"],0)
        self.assertAlmostEqual(r["inference"]["standard_error"],math.sqrt(16/3+16/3))
    def test_threshold_and_exclusions(self):
        r=self.case("no_8x8dct,Performance-\n0,9\n0,11\n1,3\n1,5\n2,99\n1,bad\n",5)
        self.assertTrue(r["practical"]["point_estimate_meets"]); self.assertEqual(r["excluded"],2)
    def test_undefined_cases_and_probability(self):
        r=self.case("no_8x8dct,Performance-\n0,1\n1,2\n"); self.assertIsNone(r["inference"])
        r=self.case("no_8x8dct,Performance-\n0,1\n0,1\n1,1\n1,1\n"); self.assertIsNone(r["inference"])
        self.assertAlmostEqual(_t_cdf(0,1),.5); self.assertGreater(_t_cdf(2,10),.9); self.assertLess(_t_cdf(-2,10),.1)
