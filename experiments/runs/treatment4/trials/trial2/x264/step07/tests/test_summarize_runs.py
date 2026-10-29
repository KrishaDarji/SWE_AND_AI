import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarize_runs import (CountedEvaluator, IncrementalFlow, blame, certify, evaluate, flow,
                            justify, load_dataset, predict, select_budget, summarize)


class SummarizeRunsTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkstemp(suffix=".csv")[1])
        self.path.write_text(
            "a,b,Performance-\n1,0,10.0\n0,1,20.0\n1,1,15.0\n", encoding="utf-8"
        )

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_loads_numeric_settings_and_measure(self):
        names, x, y = load_dataset(self.path)
        self.assertEqual(names, ["a", "b"])
        np.testing.assert_array_equal(x, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(y, [10, 20, 15])

    def test_summary_has_expected_statistics_and_extrema(self):
        result = summarize(self.path)
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["setting_count"], 2)
        self.assertEqual(result["best_run"]["value"], 10.0)
        self.assertEqual(result["worst_run"]["value"], 20.0)
        self.assertAlmostEqual(result["measure_summary"]["mean"], 15.0)
        self.assertEqual(set(result["setting_on_counts"]), {"a", "b"})

    def test_grouped_evaluation_is_reproducible_and_compares_with_mean_baseline(self):
        result = evaluate(self.path, test_size=0.34, random_state=7)
        self.assertEqual(result, evaluate(self.path, test_size=0.34, random_state=7))
        self.assertGreater(result["train_groups"], 0)
        self.assertGreater(result["test_groups"], 0)
        self.assertLessEqual(result["mae"], result["baseline_mae"])

    def test_predict_accepts_a_complete_configuration(self):
        value = predict(self.path, {"a": 1, "b": 0})
        self.assertIsInstance(value, float)

    def test_predict_rejects_incomplete_configuration(self):
        with self.assertRaises(ValueError):
            predict(self.path, {"a": 1})

    def test_certification_uses_predeclared_no_asm_groups(self):
        path = self.path
        path.write_text("no_8x8dct,other,Performance-\n0,0,10\n0,1,11\n1,0,20\n1,1,21\n", encoding="utf-8")
        result = certify(path, practical_threshold=5, permutations=100, random_state=3)
        self.assertEqual(result["group_definition"], {"A": "no_8x8dct=0", "B": "no_8x8dct=1"})
        self.assertEqual((result["group_a_n"], result["group_b_n"]), (2, 2))
        self.assertEqual(result["observed_difference_a_minus_b"], -10.0)
        self.assertTrue(result["point_estimate_exceeds_threshold"])
        self.assertTrue(result["confidence_interval_entirely_beyond_threshold"])
        self.assertEqual(result, certify(path, practical_threshold=5, permutations=100, random_state=3))

    def test_welch_interval_uses_unequal_variance_degrees_of_freedom(self):
        self.path.write_text("no_8x8dct,other,Performance-\n0,0,0\n0,1,10\n1,0,0\n1,1,-2\n", encoding="utf-8")
        result = certify(self.path, practical_threshold=5, permutations=100, random_state=4)
        a, b = np.array([0., 10.]), np.array([0., -2.])
        va, vb = a.var(ddof=1) / 2, b.var(ddof=1) / 2
        expected_df = (va + vb) ** 2 / (va ** 2 + vb ** 2)
        self.assertAlmostEqual(result["welch_satterthwaite_df"], expected_df)
        self.assertAlmostEqual(result["difference_ci_95"][0], stats.t.interval(
            .95, expected_df, loc=6, scale=np.sqrt(va + vb))[0])
        self.assertTrue(result["point_estimate_exceeds_threshold"])
        self.assertFalse(result["confidence_interval_entirely_beyond_threshold"])
        self.assertEqual(result["practical_importance"], "uncertain: interval crosses practical threshold")

    def test_incremental_prefixes_match_batch_statistics(self):
        state = IncrementalFlow(["a", "b"])
        rows = [{"a": 1, "b": 0, "Performance-": 10},
                {"a": 0, "b": 1, "Performance-": 20},
                {"a": 1, "b": 1, "Performance-": 15}]
        for index, row in enumerate(rows, 1):
            snapshot = state.update(row)
            values = np.array([r["Performance-"] for r in rows[:index]], dtype=float)
            self.assertEqual(snapshot["count"], index)
            self.assertAlmostEqual(snapshot["mean"], values.mean())
            self.assertAlmostEqual(snapshot["sample_variance"], values.var(ddof=1)) if index > 1 else self.assertIsNone(snapshot["sample_variance"])

    def test_empty_one_observation_and_tied_best(self):
        state = IncrementalFlow(["a"])
        self.assertEqual(state.snapshot(), {"count": 0, "mean": None, "sample_variance": None, "best": None})
        self.assertEqual(state.update({"a": 1, "Performance-": 4})["best"]["value"], 4)
        state.update({"a": 0, "Performance-": 4})
        self.assertEqual(state.snapshot()["best"]["settings"], {"a": 1.0})

    def test_flow_empty_input(self):
        self.path.write_text("a,Performance-\n", encoding="utf-8")
        self.assertEqual(flow(self.path), [])

    def test_blame_known_associations_constants_multivalue_and_order(self):
        self.path.write_text(
            "slow,constant,multi,other,Performance-\n"
            "0,7,0,0,10\n0,7,1,1,11\n1,7,0,1,30\n1,7,2,0,31\n", encoding="utf-8")
        result = blame(self.path)
        self.assertEqual(set(item["setting"] for item in result["settings"]), {"slow", "multi", "other", "constant"})
        self.assertEqual(result["settings"][-1]["setting"], "other")
        by_name = {item["setting"]: item for item in result["settings"]}
        self.assertTrue(by_name["constant"]["constant"])
        self.assertEqual(by_name["multi"]["level_count"], 3)
        self.assertEqual([g["n"] for g in by_name["multi"]["levels"]], [2, 1, 1])
        self.assertGreater(by_name["slow"]["eta_squared"], by_name["other"]["eta_squared"])

    def test_blame_detects_mutually_exclusive_settings_and_ties_are_deterministic(self):
        self.path.write_text("a,b,Performance-\n1,0,1\n0,1,1\n1,0,3\n0,1,3\n", encoding="utf-8")
        result = blame(self.path)
        self.assertEqual(result["mutually_exclusive_pairs"], [["a", "b"]])
        self.assertEqual([x["setting"] for x in result["settings"]], ["a", "b"])

    def test_justify_selects_lowest_observed_and_reports_reference_and_evidence(self):
        self.path.write_text(
            "a,b,Performance-\n0,0,8\n1,0,3\n0,1,3\n1,1,9\n", encoding="utf-8"
        )
        result = justify(self.path)
        self.assertEqual(result["recommendation"]["record_id"], "csv_record_2")
        self.assertEqual(result["recommendation"]["csv_row"], 3)
        self.assertEqual(result["recommendation"]["configuration"], {"a": 1.0, "b": 0.0})
        self.assertEqual(result["recommendation"]["measured_performance"], 3.0)
        self.assertEqual(result["historical_reference"]["record_id"], "csv_record_1")
        self.assertEqual(result["measured_comparison"]["reference_minus_recommendation"], 5.0)
        self.assertTrue(any("Associations do not establish" in note for note in result["evidence_notes"]))

    def test_justify_keeps_measured_estimate_and_association_categories_distinct(self):
        self.path.write_text("a,b,Performance-\n0,0,10\n1,0,4\n0,1,7\n1,1,5\n", encoding="utf-8")
        result = justify(self.path)
        recommendation = result["recommendation"]
        self.assertIn("measured_performance", recommendation)
        self.assertIn("model_estimate", recommendation)
        self.assertIn("association_evidence", result)
        self.assertNotEqual(result["association_evidence"]["measure"], "model_estimate")

    def test_budget_is_enforced_and_recommendation_is_evaluated(self):
        candidates = [{"a": 0}, {"a": 1}, {"a": 2}]
        evaluator = CountedEvaluator({(0.,): 9, (1.,): 2, (2.,): 4}, ["a"], 2)
        result = select_budget(candidates, evaluator, 2, 42)
        self.assertEqual(result["budget_used"], 2)
        self.assertIn(result["recommendation"], result["observations"])
        self.assertEqual(len({tuple(x["configuration"].values()) for x in result["observations"]}), 2)

    def test_reproducibility_and_no_duplicate_evaluations(self):
        candidates = [{"a": i} for i in range(10)]
        def run():
            ev = CountedEvaluator({(float(i),): float(i) for i in range(10)}, ["a"], 20)
            return select_budget(candidates, ev, 20, 42)
        self.assertEqual(run(), run())

    def test_invalid_and_zero_budgets_are_rejected(self):
        with self.assertRaises(ValueError):
            CountedEvaluator({}, ["a"], 0)
        ev = CountedEvaluator({(0.,): 1}, ["a"], 1)
        with self.assertRaises(ValueError):
            select_budget([], ev, 1)

    def test_outcome_isolation_and_budget_overrun(self):
        class Hidden:
            def __init__(self): self.calls = []
            def evaluate(self, configuration):
                self.calls.append(configuration)
                return 3.0
        hidden = Hidden()
        result = select_budget([{"a": 0}, {"a": 1}], hidden, 2, 42)
        self.assertEqual(len(hidden.calls), 2)
        self.assertNotIn("outcomes", result)
        limited = CountedEvaluator({(0.,): 1, (1.,): 2}, ["a"], 1)
        limited.evaluate({"a": 0})
        with self.assertRaises(RuntimeError):
            limited.evaluate({"a": 1})


if __name__ == "__main__":
    unittest.main()
