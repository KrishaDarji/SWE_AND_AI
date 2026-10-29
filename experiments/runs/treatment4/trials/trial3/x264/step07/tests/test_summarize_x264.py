import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_x264", ROOT / "source" / "summarize_x264.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_summary_frequencies_missing_and_outcome(self):
        path = self.write("a,b,Performance-\n1,0,10\n1,0,20\n0,1,\n")
        got = module.summarize(path)
        self.assertEqual(got["record_count"], 3)
        self.assertEqual(got["setting_frequencies"]["a"], {"0": 1, "1": 2})
        self.assertEqual(got["missing_values"]["Performance-"], 1)
        self.assertEqual(got["outcome"], {"name": "Performance-", "direction": "lower is better"})
        self.assertEqual(got["numeric_statistics"]["Performance-"]["count"], 2)

    def test_empty_input(self):
        path = self.write("")
        got = module.summarize(path)
        self.assertEqual(got["record_count"], 0)
        self.assertEqual(got["encoder_settings"], [])

    def test_all_missing_outcome(self):
        path = self.write("x,Performance-\n1,not-a-number\n")
        got = module.summarize(path)
        self.assertEqual(got["numeric_statistics"]["Performance-"], {"count": 0})

    def test_grouped_evaluation_is_reproducible_and_reports_settings(self):
        path = self.write("a,b,Performance-\n0,x,1\n0,x,2\n1,y,10\n1,y,11\n2,z,20\n2,z,21\n3,w,30\n3,w,31\n")
        one = module.evaluate(path); two = module.evaluate(path)
        self.assertEqual(one["predictor_names"], ["a", "b"])
        self.assertEqual((one["train_count"], one["test_count"]), (6, 2))
        self.assertEqual(one["held_out_mae"], two["held_out_mae"])
        self.assertIn("GroupShuffleSplit", one["model_settings"]["group_split"])

    def test_prediction_missing_setting_and_invalid_target(self):
        path = self.write("a,b,Performance-\n0,x,1\n1,y,2\n2,z,bad\n3,w,4\n4,q,5\n")
        result = module.evaluate(path)
        self.assertEqual(result["invalid_target_rows"], 1)
        self.assertEqual(len(module.predict(path, [{"a": "9", "b": "new"}])), 1)
        with self.assertRaises(ValueError):
            module.predict(path, [{"a": "9"}])

    def test_training_only_fit_does_not_use_held_out_outcomes(self):
        first = self.write("a,Performance-\n0,1\n0,2\n1,10\n1,11\n2,20\n2,21\n3,30\n3,31\n4,40\n4,41\n")
        a = module.evaluate(first)
        held_out = {key.split("\x1f")[0] for key in a["test_groups"]}
        second = self.write("a,Performance-\n" + "".join(
            f"{i},{(10000 + i * 100 + j) if str(i) in held_out else (i * 10 + j)}\n"
            for i in range(5) for j in (0, 1)))
        b = module.evaluate(second)
        self.assertEqual(a["training_mean"], module.evaluate(first)["training_mean"])
        self.assertEqual(module.predict(first, [{"a": "0"}]), module.predict(first, [{"a": "0"}]))

    def test_setting_missingness_is_normalized_and_imputed_from_training(self):
        path = self.write("a,b,Performance-\n1,,1\nNaN,2,2\ninf,2,3\n1,2,4\n")
        result = module.evaluate(path)
        self.assertEqual(module._setting(""), None)
        self.assertEqual(module._setting("not-a-number"), None)
        self.assertEqual(module._setting("NaN"), None)
        self.assertEqual(module._setting("inf"), None)
        imputer = result["model"].named_steps["preprocess"].named_transformers_["settings"].named_steps["impute"]
        self.assertEqual(imputer.statistics_.tolist(), ["1.0", "2.0"])
        transformed = result["model"].named_steps["preprocess"].transform([[None, None]])
        self.assertEqual(transformed.shape[0], 1)

    def test_completely_missing_training_setting_uses_explicit_marker(self):
        path = self.write("a,b,Performance-\n,1,1\n,2,2\n,3,3\n,4,4\n,5,5\n")
        result = module.evaluate(path)
        imputer = result["model"].named_steps["preprocess"].named_transformers_["settings"].named_steps["impute"]
        self.assertEqual(imputer.statistics_[0], module.MISSING_SETTING)
        self.assertEqual(module.predict(path, [{"a": "NaN", "b": "6"}])[0] > 0, True)

    def test_certify_direction_and_unequal_variance_interval(self):
        path = self.write("no_8x8dct,Performance-\n0,10\n0,12\n0,14\n1,20\n1,21\n1,22\n")
        got = module.certify(path, 5)
        self.assertEqual((got["comparison"]["A"]["n"], got["comparison"]["B"]["n"]), (3, 3))
        self.assertEqual(got["mean_difference_A_minus_B"], -9.0)
        self.assertNotEqual(got["welch"]["df"], 4.0)
        self.assertLess(got["welch"]["interval_A_minus_B"][0], -9)
        self.assertGreater(got["welch"]["interval_A_minus_B"][1], -9)

    def test_certify_missing_values_and_threshold_crossing(self):
        path = self.write("no_8x8dct,Performance-\n0,0\n0,10\n0,20\n1,0\n1,1\n1,2\n1,NaN\n")
        got = module.certify(path, 5)
        self.assertEqual(got["comparison"]["A"]["n"], 3)
        self.assertTrue(got["practical_effect"]["point_estimate_magnitude_exceeds_threshold"])
        self.assertEqual(got["practical_effect"]["favored_group"], "uncertain")

    def test_certify_positive_difference_favors_b_and_default_provenance(self):
        path = self.write("no_8x8dct,Performance-\n0,20\n0,21\n0,22\n1,10\n1,11\n1,12\n")
        got = module.certify(path)
        self.assertEqual(got["practical_effect"]["favored_group"], "B")
        self.assertTrue(got["practical_effect"]["interval_entirely_above_positive_threshold"])
        self.assertEqual(got["practical_effect"]["threshold_source"], "demonstration_default")

    def test_certify_negative_difference_favors_a_and_crosses_zero(self):
        path = self.write("no_8x8dct,Performance-\n0,10\n0,11\n0,12\n1,20\n1,21\n1,22\n")
        got = module.certify(path, 5)
        self.assertEqual(got["practical_effect"]["favored_group"], "A")
        self.assertTrue(got["practical_effect"]["interval_entirely_below_negative_threshold"])
        self.assertEqual(got["practical_effect"]["threshold_source"], "supplied")
        crossing = module.certify(self.write("no_8x8dct,Performance-\n0,10\n0,30\n0,50\n1,20\n1,30\n1,40\n"), 5)
        self.assertEqual(crossing["practical_effect"]["favored_group"], "uncertain")
        self.assertTrue(crossing["welch"]["interval_A_minus_B"][0] < 0 < crossing["welch"]["interval_A_minus_B"][1])

    def test_certify_invalid_threshold_insufficient_and_zero_variance(self):
        with self.assertRaises(ValueError):
            module.certify(self.write("no_8x8dct,Performance-\n0,1\n0,2\n1,3\n1,4\n"), -1)
        with self.assertRaises(ValueError):
            module.certify(self.write("no_8x8dct,Performance-\n0,1\n1,2\n"))
        with self.assertRaises(ValueError):
            module.certify(self.write("no_8x8dct,Performance-\n0,1\n0,1\n1,2\n1,3\n"))

    def test_incremental_prefixes_match_batch_statistics_and_retain_configuration(self):
        records = [{"a": 1, "Performance-": "10"}, {"a": 2, "Performance-": "6"},
                   {"a": 3, "Performance-": "6"}, {"a": 4, "Performance-": "bad"}]
        snapshots = list(module.benchmark_flow(iter(records), ["a"]))
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(snapshots[0]["sample_variance"], None)
        self.assertAlmostEqual(snapshots[2]["mean"], sum((10, 6, 6)) / 3)
        self.assertAlmostEqual(snapshots[2]["sample_variance"], 16 / 3)
        self.assertEqual(snapshots[2]["best"]["record_id"], 1)
        self.assertEqual(snapshots[2]["best"]["configuration"], {"a": 2})
        self.assertEqual(snapshots[3]["count"], 3)

    def test_empty_ties_defensive_copy_and_mutating_input(self):
        self.assertEqual(list(module.benchmark_flow(iter([]))), [])
        record = {"cfg": {"x": 1}, "Performance-": 2}
        first = next(module.benchmark_flow(iter([record]), ["cfg"]))
        record["cfg"]["x"] = 99
        self.assertEqual(first["best"]["configuration"], {"cfg": {"x": 1}})
        tied = list(module.benchmark_flow(iter([{"Performance-": 1}, {"Performance-": 1}])))[-1]
        self.assertEqual(tied["best"]["record_id"], 0)

    def test_csv_integration_and_iterator_is_requested_incrementally(self):
        class Guard:
            def __init__(self): self.i = 0
            def __iter__(self): return self
            def __next__(self):
                if self.i == 0:
                    self.i += 1; return {"a": "1", "Performance-": "5"}
                if self.i == 1:
                    self.i += 1; return {"a": "2", "Performance-": "3"}
                raise AssertionError("requested beyond the demonstrated prefix")
        stream = module.benchmark_flow(Guard())
        self.assertEqual(next(stream)["count"], 1)
        self.assertEqual(next(stream)["count"], 2)
        path = self.write("a,Performance-\n1,5\n2,3\n")
        self.assertEqual(list(module.benchmark_csv(path))[-1]["best"]["outcome"], 3.0)

    def test_blame_known_association_constants_missing_and_deterministic_ties(self):
        path = self.write("a,b,c,d,Performance-\n0,0,1,,1\n0,0,1,,2\n1,1,1,,10\n1,1,1,,11\n0,0,1,,10\n0,0,1,,11\n")
        got = module.blame(path)
        self.assertEqual([item["setting"] for item in got["ranking"]], ["a", "b", "c", "d"])
        self.assertAlmostEqual(got["ranking"][0]["primary_score"], got["ranking"][1]["primary_score"])
        self.assertEqual(got["ranking"][2]["primary_score"], 0.0)
        self.assertEqual(got["ranking"][0]["groups"][0]["count"], 4)
        self.assertEqual(got["ranking"][0]["fastest_group"]["value"], "0.0")

    def test_blame_small_groups_and_invalid_target(self):
        path = self.write("a,b,Performance-\n0,x,1\n1,x,2\n2,y,bad\n")
        got = module.blame(path)
        self.assertEqual(got["invalid_target_rows"], 1)
        self.assertEqual(got["ranking"][0]["score_status"], "undefined; reported as 0.0")

    def test_justify_selects_lowest_and_first_record_arithmetic(self):
        path = self.write("a,b,Performance-\nold,x,10\nbest,y,3\ntie,z,3\nbad,q,NaN\n")
        got = module.justify(path)
        self.assertEqual(got["recommendation"]["record_id"], 1)
        self.assertEqual(got["recommendation"]["configuration"], {"a": "best", "b": "y"})
        self.assertEqual(got["historical_comparison"]["recommended_minus_reference"], -7.0)
        self.assertEqual(got["invalid_or_nonfinite_outcome_count"], 1)
        self.assertEqual(got["recommendation"]["evidence_label"], "measured observed dataset record")

    def test_justify_missing_reference_and_no_usable_records(self):
        missing = self.write("a,Performance-\nfirst,NaN\nb,2\n")
        self.assertIsNone(module.justify(missing)["historical_comparison"])
        empty = self.write("a,Performance-\nx,inf\ny,bad\n")
        with self.assertRaisesRegex(ValueError, "no usable records"):
            module.justify(empty)

    def test_choose_budget_reproducibility_duplicates_and_hidden_outcomes(self):
        candidates = [{"a": str(i)} for i in range(30)] + [{"a": "3"}]
        outcomes = {str(i): float(100 - i) for i in range(30)}
        calls = []
        def guarded(configuration):
            calls.append(configuration)
            return outcomes[configuration["a"]]
        one = module.choose_configuration(candidates, guarded, 20, 42)
        calls2 = []
        two = module.choose_configuration(candidates, lambda c: (calls2.append(c), outcomes[c["a"]])[1], 20, 42)
        self.assertEqual(one, two)
        self.assertEqual(len(calls), 20)
        self.assertEqual(len({module._configuration_key(c) for c in calls}), 20)
        self.assertIn(one["recommendation"]["configuration"], calls)
        self.assertEqual(one["budget_used"], 20)

    def test_choose_validation_and_budget_exceeding_candidates(self):
        candidates = [{"a": "x"}, {"a": "x"}, {"a": "y"}]
        got = module.choose_configuration(candidates, lambda c: 2 if c["a"] == "x" else 1, 20)
        self.assertEqual(got["budget_used"], 2)
        self.assertEqual(got["recommendation"]["configuration"], {"a": "y"})
        for budget in (0, -1, True):
            with self.assertRaises(ValueError):
                module.choose_configuration(candidates, lambda c: 1, budget)
        with self.assertRaises(ValueError):
            module.choose_configuration([], lambda c: 1)

    def test_choose_rejects_nonfinite_callback_outcome_and_audit_is_separate(self):
        with self.assertRaises(ValueError):
            module.choose_configuration([{"a": 1}], lambda c: float("nan"))
        self.assertEqual(module.audit_observed_minimum([{"a": 1, "Performance-": 3},
                                                        {"a": 2, "Performance-": 2}])["observed_minimum"]["performance"], 2)


if __name__ == "__main__":
    unittest.main()
