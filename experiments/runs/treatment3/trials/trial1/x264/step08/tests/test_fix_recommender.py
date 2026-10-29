import csv
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from fix_recommender import load_data, recommend


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(self.file)
        writer.writerow(["a", "b", "Performance-"])
        writer.writerows([[0, 0, 10], [1, 0, 5], [0, 1, 8], [1, 1, 9]])
        self.file.close()

    def tearDown(self):
        os.unlink(self.file.name)

    def test_load_and_one_change(self):
        names, X, y = load_data(self.file.name)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(X.shape, (4, 2))
        result = recommend(self.file.name, {"a": 0, "b": 0})
        self.assertEqual(result["setting_changed"], "a")
        self.assertEqual(result["from"], 0)
        self.assertEqual(result["to"], 1)

    def test_unknown_setting_rejected(self):
        with self.assertRaises(ValueError):
            recommend(self.file.name, {"missing": 1})


if __name__ == "__main__":
    unittest.main()
