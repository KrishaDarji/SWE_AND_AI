import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import fix_x264


class FixTests(unittest.TestCase):
    def test_real_dataset_has_one_setting_improvement(self):
        workspace = os.path.abspath(os.path.join(HERE, *("..",) * 8))
        fields, settings, rows = fix_x264.load_dataset(
            os.path.join(workspace, "moot", "optimize", "config", "X264_AllMeasurements.csv")
        )
        baseline, fix = fix_x264.find_fix(rows, settings)
        self.assertIsNotNone(fix)
        changed, revised = fix
        self.assertEqual(fix_x264.hamming_distance(baseline, revised, settings), 1)
        self.assertLess(revised[fix_x264.METRIC], baseline[fix_x264.METRIC])
        self.assertIn(changed, settings)

    def test_no_improvement_is_supported(self):
        rows = [{"a": "0", "Performance-": 3.0}]
        baseline, fix = fix_x264.find_fix(rows, ["a"])
        self.assertEqual(baseline["Performance-"], 3.0)
        self.assertIsNone(fix)

    def test_validation_rejects_non_binary_setting(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as handle:
            handle.write("a,Performance-\n2,1\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                fix_x264.load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
