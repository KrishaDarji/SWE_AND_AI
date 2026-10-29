import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from certify import certify, percentile


class CertifyTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3)

    def test_clear_improvement_is_credible(self):
        result = certify([10] * 20 + [1] * 5, repetitions=300)
        self.assertTrue(result["credible"])
        self.assertGreater(result["improvement"], 0)

    def test_rejects_small_input(self):
        with self.assertRaises(ValueError):
            certify([1, 2, 3])


if __name__ == "__main__":
    unittest.main()
