import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_quality_summary import (IncrementalEvaluator, blame, certify, classify_columns,
                                   predict, summarize, summarize_column)


class SummaryTests(unittest.TestCase):
    def test_classification_uses_metric_suffix(self):
        self.assertEqual(classify_columns(["MRE-", "ACC+", "depth"]), (["MRE-", "ACC+"], ["depth"]))

    def test_numeric_and_categorical_statistics(self):
        numeric = summarize_column("score", ["1", "3", "2"])
        categorical = summarize_column("criterion", ["a", "b", "a"])
        self.assertEqual((numeric.minimum, numeric.maximum, numeric.mean, numeric.median), (1.0, 3.0, 2.0, 2.0))
        self.assertEqual((categorical.mode, categorical.values), ("a", {"a": 2, "b": 1}))

    def test_end_to_end_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,depth,criterion\n0.2,0.8,3,a\n0.4,0.6,5,b\n", encoding="utf-8")
            result = summarize(path)
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["metric_fields"], ["MRE-", "ACC+"])
            self.assertEqual(result["columns"][3]["mode"], "a")

    def test_prediction_uses_nearest_configuration_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,depth,criterion\n0.2,0.8,3,a\n0.8,0.4,10,b\n", encoding="utf-8")
            result = predict(path, {"depth": 3, "criterion": "a"}, neighbors=1)
            self.assertEqual((result["predicted_error"], result["predicted_accuracy"]), (0.2, 0.8))

    def test_prediction_rejects_invalid_neighbor_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,depth\n0.2,0.8,3\n", encoding="utf-8")
            with self.assertRaises(ValueError): predict(path, {"depth": 3}, neighbors=0)

    def test_certification_identifies_reliable_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,family\n0.1,0.9,A\n0.2,0.8,A\n0.8,0.2,B\n0.7,0.3,B\n", encoding="utf-8")
            self.assertEqual(certify(path, "family", "A", "B", resamples=300, seed=3)["reliable_winner"], "A")

    def test_certification_requires_both_families(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,family\n0.1,A\n", encoding="utf-8")
            with self.assertRaises(ValueError): certify(path, "family", "A", "B")

    def test_incremental_evaluation_updates_best_as_results_arrive(self):
        evaluator = IncrementalEvaluator(["MRE-", "ACC+"])
        first = evaluator.submit({"depth": 10}, {"MRE-": "0.4", "ACC+": "0.7"})
        self.assertTrue(first["is_best"])
        second = evaluator.submit({"depth": 5}, {"MRE-": "0.2", "ACC+": "0.9"})
        self.assertTrue(second["is_best"])
        third = evaluator.submit({"depth": 2}, {"MRE-": "0.3", "ACC+": "0.95"})
        self.assertFalse(third["is_best"])
        self.assertEqual(evaluator.report()["best"]["configuration"], {"depth": 5})

    def test_incremental_evaluation_rejects_missing_validation_result(self):
        evaluator = IncrementalEvaluator(["MRE-", "ACC+"])
        with self.assertRaises(ValueError): evaluator.submit({}, {"MRE-": "0.2"})

    def test_blame_ranks_values_associated_with_high_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,depth,criterion\n0.1,0.9,3,a\n0.9,0.1,3,b\n0.8,0.2,5,b\n0.2,0.8,5,a\n", encoding="utf-8")
            result = blame(path, quantile=0.75)
            self.assertEqual(result["error_field"], "MRE-")
            self.assertEqual(result["parameters"]["criterion"][0]["value"], "b")
            self.assertGreater(result["parameters"]["criterion"][0]["high_error_rate"], 0)

    def test_blame_rejects_unknown_metric_and_bad_quantile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,depth\n0.2,3\n", encoding="utf-8")
            with self.assertRaises(ValueError): blame(path, "missing")
            with self.assertRaises(ValueError): blame(path, quantile=0)


if __name__ == "__main__":
    unittest.main()
