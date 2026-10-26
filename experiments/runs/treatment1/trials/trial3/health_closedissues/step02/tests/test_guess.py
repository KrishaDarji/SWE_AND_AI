import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from guess import encode, load_rows, predict_from_rows, validation_summary


class GuessTests(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.getcwd(), "moot/optimize/hpo/Health-ClosedIssues0000.csv")
        self.rows = load_rows(path)

    def test_dataset_and_encoding(self):
        self.assertEqual(len(self.rows), 10000)
        self.assertEqual(len(encode(self.rows[0])), 5)

    def test_prediction_is_bounded(self):
        pred = predict_from_rows(self.rows[1:50], self.rows[0])
        self.assertTrue(0 <= pred["MRE-"] <= 2)
        self.assertTrue(0 <= pred["ACC+"] <= 1)

    def test_validation_is_deterministic(self):
        self.assertEqual(validation_summary(self.rows[:100], seed=7), validation_summary(self.rows[:100], seed=7))


if __name__ == "__main__":
    unittest.main()
