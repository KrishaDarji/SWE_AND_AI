import csv
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "source"))
import blame


DATA = HERE.parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class BlameTests(unittest.TestCase):
    def test_ranks_use_average_ties(self):
        self.assertEqual(blame.ranks([2, 1, 2]), [2.5, 1.0, 2.5])

    def test_constant_characteristics_are_ignored(self):
        rows, result = blame.analyze(DATA)
        self.assertEqual(len(rows), 93)
        self.assertNotIn("prec", {item["name"] for item in result})

    def test_outputs_are_not_treated_as_drivers(self):
        _, result = blame.analyze(DATA)
        self.assertTrue({item["name"] for item in result}.isdisjoint(blame.OUTPUTS))

    def test_known_driver_is_reported(self):
        _, result = blame.analyze(DATA)
        self.assertIn("cplx", {item["name"] for item in result})


if __name__ == "__main__":
    unittest.main()
