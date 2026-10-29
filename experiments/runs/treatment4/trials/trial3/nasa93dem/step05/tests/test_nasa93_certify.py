import math
import unittest
from source.nasa93_certify import certify

class CertifyTests(unittest.TestCase):
    def rows(self, a, b):
        return [{"tool": "n", "Effort-": x} for x in a] + [{"tool": "h", "Effort-": x} for x in b]

    def test_direction_and_counts(self):
        r = certify(self.rows([20, 22, 24], [10, 12, 14]), 1)
        self.assertEqual((r["n_a"], r["n_b"]), (3, 3)); self.assertGreater(r["difference_a_minus_b"], 0)
        self.assertEqual(r["practical_conclusion"], "favors B")

    def test_welch_arithmetic_and_negative_direction(self):
        r = certify(self.rows([1, 2, 3], [10, 20, 30, 40]), 0)
        self.assertAlmostEqual(r["difference_a_minus_b"], -23.0)
        self.assertAlmostEqual(r["degrees_of_freedom"], 3.0478994016574417, places=10)
        self.assertLess(r["confidence_interval_95"][1], 0)
        self.assertEqual(r["practical_conclusion"], "favors A")

    def test_threshold_crossing_and_no_significance_required(self):
        r = certify(self.rows([0, 100], [0, 0]), 40)
        self.assertTrue(r["point_exceeds_threshold"]); self.assertEqual(r["practical_conclusion"], "uncertain")

    def test_missing_insufficient_zero_variance_and_invalid_threshold(self):
        with self.assertRaises(ValueError): certify(self.rows([1], [2, 3]), 1)
        with self.assertRaises(ValueError): certify(self.rows([1, 1], [2, 2]), 1)
        with self.assertRaises(ValueError): certify(self.rows([1, None, float("nan")], [2, 3]), 1)
        with self.assertRaises(ValueError): certify(self.rows([1, 2], [3, 4]), -1)

if __name__ == "__main__": unittest.main()
