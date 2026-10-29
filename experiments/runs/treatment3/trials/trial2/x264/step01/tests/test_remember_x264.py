import os
import tempfile
import unittest

from source.remember_x264 import analyze, identify_columns, load_dataset


DATASET = os.path.join(os.path.dirname(__file__), "../../../../../../../moot/optimize/config/X264_AllMeasurements.csv")


class RememberX264Tests(unittest.TestCase):
    def test_real_dataset_loads_and_columns_are_identified(self):
        fields, rows = load_dataset(os.path.abspath(DATASET))
        config, performance = identify_columns(fields)
        self.assertEqual(len(rows), 1152)
        self.assertEqual(len(config), 16)
        self.assertEqual(performance, ["Performance-"])

    def test_analysis_contains_measured_sections(self):
        result = analyze(os.path.abspath(DATASET))
        self.assertIn("rows=1152 columns=17", result)
        self.assertIn("performance_summary=", result)
        self.assertIn("linear_regression train_r2=", result)
        self.assertIn("data_quality=", result)

    def test_validation_rejects_blank_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("setting,Performance-\n1,\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
