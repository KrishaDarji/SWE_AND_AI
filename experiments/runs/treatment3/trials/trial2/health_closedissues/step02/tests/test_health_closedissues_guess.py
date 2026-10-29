import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from health_closedissues_guess import FEATURES, TARGETS, evaluate, load_dataset

DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv"))


class GuessTests(unittest.TestCase):
    def test_load_and_schema(self):
        x, y = load_dataset(DATASET)
        self.assertEqual(len(x), len(y))
        self.assertGreater(len(x), 4)
        self.assertEqual(set(x[0]), set(FEATURES))
        self.assertEqual(set(y[0]), set(TARGETS))

    def test_evaluation_has_finite_metrics(self):
        result = evaluate(DATASET)
        self.assertGreater(result["test_rows"], 0)
        for metric in result["metrics"].values():
            for value in metric.values():
                self.assertTrue(value == value)

    def test_validation_rejects_missing_column(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.writer(handle)
            writer.writerow(["bad"])
            writer.writerow(["1"])
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
