import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source.fix import suggest_fix


class FixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "measurements.csv")
        with open(self.path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["a", "b", "Performance-"])
            writer.writerow([0, 0, 10])
            writer.writerow([1, 0, 14])
            writer.writerow([0, 1, 12])
            writer.writerow([1, 1, 13])

    def tearDown(self):
        self.directory.cleanup()

    def test_best_one_setting_improvement(self):
        result = suggest_fix(self.path, {"a": 0, "b": 0})
        self.assertTrue(result["improvement"])
        self.assertEqual(result["setting"], "a")
        self.assertEqual(result["to"], 1)
        self.assertEqual(result["gain"], 4.0)

    def test_no_improvement(self):
        result = suggest_fix(self.path, {"a": 1, "b": 0})
        self.assertFalse(result["improvement"])

    def test_requires_exact_configuration(self):
        with self.assertRaises(ValueError):
            suggest_fix(self.path, {"a": 0})


if __name__ == "__main__":
    unittest.main()
