import csv, os, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import summarize, split_groups, NearestNeighborModel, load

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
