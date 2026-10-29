import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify_x264 import certify, read_performance  # noqa: E402


class CertifyTests(unittest.TestCase):
    def test_detects_clear_worst_decile(self):
        result = certify([1.0] * 90 + [10.0] * 10, bootstrap_reps=200, permutation_reps=200)
        self.assertEqual(result["worst_n"], 10)
        self.assertGreater(result["effect_difference"], 0)
        self.assertTrue(result["credible"])

    def test_no_difference_is_not_credible(self):
        result = certify([4.0] * 100, bootstrap_reps=200, permutation_reps=200)
        self.assertEqual(result["effect_difference"], 0)
        self.assertEqual(result["effect_size_cohens_d"], float("inf"))
        self.assertFalse(result["credible"])

    def test_real_dataset_has_expected_size(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        values = read_performance(path)
        self.assertEqual(len(values), 1152)
        self.assertTrue(all(math.isfinite(value) for value in values))


if __name__ == "__main__":
    unittest.main()
