import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import bootstrap_difference, choose_levels, load_rows


class CertifyTests(unittest.TestCase):
    def test_load_and_choose_levels(self):
        path = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"
        rows = load_rows(path)
        self.assertGreater(len(rows), 90)
        self.assertEqual(len(choose_levels(rows, "tool")), 2)

    def test_bootstrap_detects_clear_gap(self):
        observed, lo, hi, certified = bootstrap_difference([1, 1, 1, 1], [10, 10, 10, 10], iterations=1000)
        self.assertLess(observed, 0)
        self.assertLess(hi, 0)
        self.assertTrue(certified)

    def test_bootstrap_is_reproducible(self):
        self.assertEqual(bootstrap_difference([1, 2, 3], [2, 3, 4], iterations=100, seed=4), bootstrap_difference([1, 2, 3], [2, 3, 4], iterations=100, seed=4))


if __name__ == "__main__":
    unittest.main()
