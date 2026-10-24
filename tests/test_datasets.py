import unittest

from src.core.datasets import DATASETS, get_dataset, read_rows, schema
from src.core.remember import rank_rows, summarize
from src.core.guess import leave_one_out_mae, predict_row
from src.core.certify import certify
from src.core.flow import stream_objective
from src.core.blame import blame
from src.core.justify import justify
from src.core.choose import choose
from src.core.fix import fix
from src.core.spot import spot
from src.core.race import race


class DatasetTests(unittest.TestCase):
    def test_all_selected_datasets_exist_and_load(self):
        for spec in DATASETS.values():
            self.assertTrue(spec.path.exists(), spec.path)
            rows = read_rows(spec, limit=2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0])

    def test_objectives_are_present(self):
        for spec in DATASETS.values():
            columns = schema(spec)["columns"]
            for objective in spec.objectives:
                self.assertIn(objective, columns)

    def test_unknown_dataset_has_helpful_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset"):
            get_dataset("missing")

    def test_step1_summary_contains_objective_statistics(self):
        for spec in DATASETS.values():
            summary = summarize(spec)
            self.assertEqual(summary["rows"], len(read_rows(spec)))
            self.assertEqual(set(summary["objectives"]), set(spec.objectives))

    def test_step1_ranking_returns_requested_rows(self):
        for spec in DATASETS.values():
            rows = read_rows(spec)
            ranked = rank_rows(spec, rows, top_k=3)
            self.assertEqual(len(ranked), 3)
            self.assertGreaterEqual(ranked[0][0], ranked[-1][0])

    def test_step2_predicts_all_objectives(self):
        for spec in DATASETS.values():
            rows = read_rows(spec, limit=12)
            prediction = predict_row(spec, rows[1:], rows[0], k=3)
            self.assertEqual(set(prediction), set(spec.objectives))
            self.assertTrue(all(value == value for value in prediction.values()))

    def test_step2_leave_one_out_returns_errors(self):
        for spec in DATASETS.values():
            rows = read_rows(spec, limit=8)
            errors = leave_one_out_mae(spec, rows, k=3)
            self.assertEqual(set(errors), set(spec.objectives))
            self.assertTrue(all(value >= 0 for value in errors.values()))

    def test_step3_respects_minimization_direction(self):
        lower_is_better = certify("Effort-", [1, 2, 1], [10, 11, 9], permutations=200)
        higher_is_better = certify("ACC+", [10, 11, 9], [1, 2, 1], permutations=200)
        self.assertGreater(lower_is_better["effect"], 0)
        self.assertGreater(higher_is_better["effect"], 0)

    def test_step3_rejects_empty_samples(self):
        with self.assertRaises(ValueError):
            certify("Performance-", [], [1, 2])

    def test_step4_streams_rows_and_records_history(self):
        for spec in DATASETS.values():
            rows = read_rows(spec, limit=30)
            result = stream_objective(spec, rows, minimum=5, tolerance=0.01, patience=2)
            self.assertGreaterEqual(result["seen"], 5)
            self.assertIn("history", result)
            self.assertTrue(all(result["history"][objective] for objective in spec.objectives))

    def test_step4_rejects_invalid_controls(self):
        spec = DATASETS["x264"]
        with self.assertRaises(ValueError):
            stream_objective(spec, read_rows(spec, limit=5), minimum=0)

    def test_step5_returns_ranked_feature_findings(self):
        for spec in DATASETS.values():
            findings = blame(spec, read_rows(spec, limit=40), top_k=5)
            self.assertLessEqual(len(findings), 5)
            self.assertTrue(all("feature" in finding for finding in findings))
            effects = [finding["effect"] for finding in findings]
            self.assertEqual(effects, sorted(effects, reverse=True))

    def test_step6_justification_contains_evidence_and_text(self):
        for spec in DATASETS.values():
            rows = read_rows(spec, limit=40)
            recommendation = rank_rows(spec, rows, top_k=1)[0][1]
            explanation = justify(spec, rows, recommendation)
            self.assertEqual(explanation["dataset"], spec.name)
            self.assertTrue(explanation["reasons"])
            self.assertTrue(explanation["text"])

    def test_step7_respects_evaluation_budget(self):
        for spec in DATASETS.values():
            result = choose(spec, read_rows(spec), budget=7)
            self.assertEqual(result["evaluated"], 7)
            self.assertLessEqual(result["evaluated"], result["available_rows"])
            self.assertIn("selected", result)

    def test_step7_rejects_invalid_budget(self):
        spec = DATASETS["x264"]
        with self.assertRaises(ValueError):
            choose(spec, read_rows(spec, limit=5), budget=0)

    def test_step8_returns_a_repair_result(self):
        for spec in DATASETS.values():
            rows = read_rows(spec, limit=30)
            candidate = rows[0]
            result = fix(spec, rows, candidate)
            self.assertEqual(result["dataset"], spec.name)
            self.assertIn("improvement", result)
            self.assertEqual(set(result["candidate"]), set(candidate))

    def test_step9_returns_ranked_anomalies(self):
        for spec in DATASETS.values():
            findings = spot(spec, read_rows(spec, limit=30), top_k=5)
            self.assertLessEqual(len(findings), 5)
            scores = [finding["score"] for finding in findings]
            self.assertEqual(scores, sorted(scores, reverse=True))
            self.assertTrue(all(score >= 0 for score in scores))

    def test_step10_races_with_equal_budget(self):
        for spec in DATASETS.values():
            result = race(spec, read_rows(spec), budget=7)
            self.assertEqual(result["budget"], 7)
            self.assertIn("shared_score", result)
            self.assertIn("baseline_score", result)

    def test_step10_rejects_invalid_budget(self):
        spec = DATASETS["x264"]
        with self.assertRaises(ValueError):
            race(spec, read_rows(spec, limit=5), budget=0)


if __name__ == "__main__":
    unittest.main()
