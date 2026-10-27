import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import (OnlineBenchmarkEvaluator, blame_slow_encodes,
                          certify_improvement, evaluate_stream,
                          justify_recommendation, predict_encoding_time,
                          select_best_configuration, summarize_csv)


class SummaryTests(unittest.TestCase):
    def test_step1_summary_and_step2_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,20\n0,1,12\n1,1,24\n", encoding="utf-8")
            self.assertEqual(summarize_csv(path)["rows"], 4)
            self.assertEqual(predict_encoding_time(path, {"a": "1", "b": "0"})["predicted_time"], 20.0)

    def test_certification_finds_credible_faster_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,20\n1,0,10\n0,0,22\n1,0,12\n0,0,19\n1,0,11\n0,0,21\n1,0,13\n0,0,20\n1,0,10\n0,0,22\n1,0,12\n", encoding="utf-8")
            result = certify_improvement(path, {"a": "0", "b": "0"}, {"a": "1", "b": "0"})
        self.assertEqual(result["pairs"], 6)
        self.assertTrue(result["credible"])
        self.assertEqual(result["p_value"], 0.015625)

    def test_certification_rejects_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,12\n", encoding="utf-8")
            result = certify_improvement(path, {"a": "0", "b": "0"}, {"a": "1", "b": "0"})
        self.assertFalse(result["credible"])
        self.assertGreater(result["p_value"], 0.05)

    def test_online_evaluator_updates_best_one_result_at_a_time(self):
        evaluator = OnlineBenchmarkEvaluator(settings=("preset", "threads"))
        first = evaluator.add_result({"preset": "fast", "threads": "1"}, 30)
        self.assertEqual(first["received"], 1)
        second = evaluator.add_result({"preset": "slow", "threads": "2"}, 12)
        self.assertEqual(second["best"]["performance"], 12.0)
        self.assertEqual([item["index"] for item in second["ranking"]], [2, 1])

    def test_stream_evaluation_accepts_json_like_records(self):
        result = evaluate_stream([
            {"configuration": {"a": "0"}, "performance": 9},
            {"a": "1", "Performance-": "7"},
        ])
        self.assertEqual(result["received"], 2)
        self.assertEqual(result["best"]["configuration"], {"a": "1"})

    def test_online_evaluator_rejects_bad_configuration(self):
        evaluator = OnlineBenchmarkEvaluator(settings=("a",))
        with self.assertRaises(ValueError):
            evaluator.add_result({"b": "1"}, 2)

    def test_blame_ranks_setting_in_slow_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,30\n0,1,11\n1,1,31\n", encoding="utf-8")
            result = blame_slow_encodes(path, 0.5)
        self.assertEqual(result["slow_rows"], 2)
        self.assertEqual(result["settings"][0]["setting"], "a")
        self.assertEqual(result["settings"][0]["mean_penalty_1_minus_0"], 20.0)
        self.assertEqual(result["settings"][0]["slow_tail_enrichment"], 0.5)

    def test_blame_rejects_invalid_quantile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,Performance-\n0,1\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                blame_slow_encodes(path, 0)

    def test_justification_recommends_fastest_observed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,7\n0,1,12\n1,1,9\n", encoding="utf-8")
            result = justify_recommendation(path)
        self.assertEqual(result["recommendation"], {"a": "1", "b": "0"})
        self.assertEqual(result["performance"], 7.0)
        self.assertEqual(result["observed_row"], 2)
        self.assertIn("lowest observed", result["rationale"])

    def test_justification_predicts_requested_unseen_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,1,30\n", encoding="utf-8")
            result = justify_recommendation(path, {"a": "1", "b": "0"})
        self.assertIsNone(result["observed_row"])
        self.assertEqual(result["selection"], "requested configuration with predicted performance")
        self.assertEqual(result["evidence_rows"], 2)

    def test_selects_fastest_configuration_meeting_quality_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,quality,Performance-\n0,70,10\n1,80,12\n0,90,8\n", encoding="utf-8")
            result = select_best_configuration(path, "quality", minimum_quality=80)
        self.assertEqual(result["recommendation"], {"a": "0"})
        self.assertEqual(result["performance"], 8.0)
        self.assertEqual(result["eligible_rows"], 2)

    def test_selects_with_quality_range_and_rejects_infeasible_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,quality,Performance-\n0,70,10\n1,80,12\n", encoding="utf-8")
            result = select_best_configuration(path, "quality", 70, 75)
            self.assertEqual(result["recommendation"], {"a": "0"})
            with self.assertRaises(ValueError):
                select_best_configuration(path, "quality", minimum_quality=95)

    def test_quality_constraint_requires_valid_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,quality,Performance-\n0,70,10\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_best_configuration(path, "quality")
            with self.assertRaises(ValueError):
                select_best_configuration(path, "quality", 80, 70)


if __name__ == "__main__":
    unittest.main()
