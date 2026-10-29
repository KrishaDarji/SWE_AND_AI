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


if __name__ == "__main__":
    unittest.main()
