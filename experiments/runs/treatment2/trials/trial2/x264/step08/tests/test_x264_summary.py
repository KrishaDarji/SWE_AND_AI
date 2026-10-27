import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import (OnlineEvaluator, blame, certify, choose, evaluate_rows,
                          improve, justify, load_dataset, predict, summarize)


REAL = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"


class CompatibilityAndCertificationTests(unittest.TestCase):
    def test_previous_summary_and_prediction_behavior(self):
        dataset = load_dataset(REAL)
        self.assertEqual(summarize(dataset)["run_count"], 1152)
        config = {name: int(float(dataset.rows[0][name])) for name in dataset.settings}
        result = predict(dataset, config)
        self.assertTrue(result["exact_match"])
        self.assertGreater(result["neighbor_count"], 0)

    def test_previous_reusable_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,quality,latency\n0,10,5\n1,20,3\n", encoding="utf-8")
            result = summarize(load_dataset(path))
            self.assertEqual(result["quality"]["quality"]["mean"], 15.0)

    def test_certification_detects_credible_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "quality"])
                for value in (10, 11, 9, 10): writer.writerow([0, value])
                for value in (20, 21, 19, 20): writer.writerow([1, value])
            result = certify(load_dataset(path))
            self.assertTrue(result["any_credible"])
            self.assertIn("feature", result["credible_improvements"])
            self.assertGreater(result["settings"]["feature"]["improvement"], 0)

    def test_real_dataset_certification_is_reported(self):
        result = certify(load_dataset(REAL))
        self.assertEqual(result["measure"], "Performance-")
        self.assertEqual(len(result["settings"]), 16)
        self.assertEqual(result["settings"]["no_asm"]["status"], "not_testable")
        self.assertEqual(result["settings"]["no_8x8dct"]["status"], "tested")


class OnlineEvaluationTests(unittest.TestCase):
    def test_results_are_processed_one_at_a_time_and_best_updates(self):
        evaluator = OnlineEvaluator(("fast", "tuned"), "encoding_time")
        first = evaluator.add_result({"fast": 0, "tuned": 0}, 10)
        self.assertEqual(first["results_seen"], 1)
        self.assertEqual(first["best"]["mean"], 10.0)
        second = evaluator.add_result({"fast": 1, "tuned": 0}, 20)
        self.assertEqual(second["configurations_seen"], 2)
        self.assertEqual(second["best"]["configuration"], {"fast": 1, "tuned": 0})
        evaluator.add_result({"fast": 1, "tuned": 0}, 24)
        self.assertEqual(evaluator.best()["mean"], 22.0)
        self.assertEqual(evaluator.best()["count"], 2)

    def test_ties_are_deterministic_and_invalid_results_rejected(self):
        evaluator = OnlineEvaluator(("a", "b"))
        evaluator.add_result({"a": 1, "b": 0}, 5)
        evaluator.add_result({"a": 0, "b": 1}, 5)
        self.assertEqual(evaluator.best()["configuration"], {"a": 0, "b": 1})
        with self.assertRaises(ValueError):
            evaluator.add_result({"a": 2, "b": 0}, 6)
        with self.assertRaises(ValueError):
            evaluator.add_result({"a": 1, "b": 0}, "not-a-number")

    def test_real_dataset_can_be_consumed_as_an_online_stream(self):
        result = evaluate_rows(load_dataset(REAL))
        self.assertEqual(result["results_seen"], 1152)
        self.assertEqual(result["configurations_seen"], 1152)
        self.assertEqual(result["best"]["mean"], 821.963)


class BlameTests(unittest.TestCase):
    def test_slow_tail_ranks_setting_that_marks_slow_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("slow_mode,other,encoding_time\n"
                            "0,0,1\n0,1,2\n1,0,10\n1,1,20\n",
                            encoding="utf-8")
            result = blame(load_dataset(path), slow_fraction=0.50)
            self.assertEqual(result["slow_count"], 2)
            self.assertEqual(result["settings"][0]["setting"], "slow_mode")
            self.assertEqual(result["settings"][0]["slow_rate_difference"], 1.0)

    def test_real_dataset_blame_has_all_settings_and_cutoff(self):
        result = blame(load_dataset(REAL))
        self.assertEqual(result["measure"], "Performance-")
        self.assertEqual(len(result["settings"]), 16)
        self.assertGreater(result["slow_count"], 0)
        self.assertEqual(sorted(item["setting"] for item in result["settings"]),
                         sorted(load_dataset(REAL).settings))


class JustificationTests(unittest.TestCase):
    def test_justification_reuses_observed_best_and_exposes_caveats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,tuned,Performance-\n0,0,10\n1,0,20\n0,1,15\n1,1,25\n",
                            encoding="utf-8")
            result = justify(load_dataset(path))
            self.assertEqual(result["decision"], "recommend")
            self.assertEqual(result["recommendation"]["configuration"], {"fast": 1, "tuned": 1})
            self.assertEqual(result["recommendation"]["mean"], 25.0)
            self.assertTrue(result["limitations"])
            self.assertIn("observed", result["evidence"][0])

    def test_real_dataset_justification_is_admin_ready(self):
        result = justify(load_dataset(REAL))
        self.assertEqual(result["measure"], "Performance-")
        self.assertEqual(result["recommendation"]["count"], 1)
        self.assertEqual(len(result["recommendation"]["configuration"]), 16)
        self.assertIn("certification", result)
        self.assertIn("slow_tail", result)


class QualityConstraintTests(unittest.TestCase):
    def test_choose_selects_best_feasible_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,quality,throughput\n"
                            "0,80,100\n1,90,90\n0,95,80\n1,70,120\n",
                            encoding="utf-8")
            result = choose(load_dataset(path), 90, "quality", "throughput")
            self.assertEqual(result["decision"], "recommend")
            self.assertEqual(result["feasible_count"], 2)
            self.assertEqual(result["recommendation"]["configuration"], {"fast": 1})
            self.assertEqual(result["recommendation"]["objective"], 90.0)

    def test_choose_reports_infeasible_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,quality\n0,80\n1,90\n", encoding="utf-8")
            result = choose(load_dataset(path), 91)
            self.assertEqual(result["decision"], "no_feasible_configuration")
            self.assertIsNone(result["recommendation"])

    def test_real_dataset_quality_constraint_preserves_configuration_shape(self):
        result = choose(load_dataset(REAL), 800)
        self.assertEqual(result["decision"], "recommend")
        self.assertEqual(result["quality_measure"], "Performance-")
        self.assertEqual(len(result["recommendation"]["configuration"]), 16)


class ImprovementTests(unittest.TestCase):
    def test_recommends_one_setting_flip_with_positive_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,tuned,Performance-\n0,0,10\n1,0,20\n0,1,15\n1,1,25\n",
                            encoding="utf-8")
            result = improve(load_dataset(path), {"fast": 0, "tuned": 0})
            self.assertEqual(result["decision"], "recommend")
            self.assertEqual(result["recommendation"]["setting"], "fast")
            self.assertEqual(result["recommendation"]["configuration"], {"fast": 1, "tuned": 0})
            self.assertEqual(result["recommendation"]["improvement"], 10.0)

    def test_reports_no_improvement_when_all_flips_are_worse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,tuned,Performance-\n0,0,30\n1,0,20\n0,1,15\n1,1,10\n",
                            encoding="utf-8")
            result = improve(load_dataset(path), {"fast": 0, "tuned": 0})
            self.assertEqual(result["decision"], "no_improvement")
            self.assertIsNone(result["recommendation"])
            self.assertEqual(result["candidate_count"], 2)

    def test_real_dataset_returns_single_setting_change(self):
        dataset = load_dataset(REAL)
        configuration = {name: int(float(dataset.rows[0][name])) for name in dataset.settings}
        result = improve(dataset, configuration)
        self.assertEqual(result["candidate_count"], len(dataset.settings))
        if result["recommendation"]:
            changed = [name for name in dataset.settings
                       if result["recommendation"]["configuration"][name] != configuration[name]]
            self.assertEqual(len(changed), 1)


if __name__ == "__main__":
    unittest.main()
