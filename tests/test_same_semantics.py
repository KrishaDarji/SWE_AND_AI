import unittest

from src.core.same import cliffs, cohen, ks, same


class SameSemanticTests(unittest.TestCase):
    def test_small_measurement_noise_is_same(self):
        left = [0.10, 0.20, 0.30, 0.40, 0.50] * 30
        right = [x + (0.005 if i % 2 else -0.005) for i, x in enumerate(left)]
        result = same(left, right)
        self.assertTrue(result["same"])

    def test_large_shift_is_not_same(self):
        left = [0.10, 0.20, 0.30, 0.40, 0.50] * 30
        right = [x + 1.0 for x in left]
        result = same(left, right)
        self.assertFalse(result["same"])
        self.assertGreater(result["cohen"], 0.35)

    def test_same_requires_all_three_tests(self):
        result = same([1, 2, 3], [1, 2, 3])
        self.assertEqual(result["cohen"], 0.0)
        self.assertEqual(result["cliffs"], 0.0)
        self.assertEqual(result["ks"], 0.0)
        self.assertTrue(result["same"])

    def test_components_are_symmetric(self):
        left = [1, 2, 3, 4]
        right = [2, 3, 4, 5]
        self.assertAlmostEqual(cohen(left, right), cohen(right, left))
        self.assertAlmostEqual(cliffs(left, right), cliffs(right, left))
        self.assertAlmostEqual(ks(left, right), ks(right, left))

    def test_empty_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            same([], [1])


if __name__ == "__main__":
    unittest.main()
