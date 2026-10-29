import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import x264_fix

DATASET = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv"

class X264FixTests(unittest.TestCase):
    def test_loads_and_validates_dataset(self):
        settings, rows = x264_fix.load_dataset(DATASET)
        self.assertEqual((16, 1152), (len(settings), len(rows)))

    def test_finds_one_setting_improvement(self):
        settings, rows = x264_fix.load_dataset(DATASET)
        recommendation, baseline, count = x264_fix.find_recommendation(settings, rows)
        self.assertIsNotNone(recommendation)
        self.assertEqual(1, sum(recommendation.baseline[x] != recommendation.candidate[x] for x in settings))
        self.assertLess(recommendation.revised_result, recommendation.baseline_result)
        self.assertGreater(count, 0)

    def test_no_improvement_case(self):
        rows = [{"a": "0", "Performance-": 2.0}, {"a": "1", "Performance-": 2.0}]
        old = x264_fix.GROUPS
        x264_fix.GROUPS = ()
        try:
            result, _, _ = x264_fix.find_recommendation(["a"], rows)
            self.assertIsNone(result)
        finally:
            x264_fix.GROUPS = old

if __name__ == "__main__":
    unittest.main()
