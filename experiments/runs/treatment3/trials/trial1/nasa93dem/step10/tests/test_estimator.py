import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from estimator import build_model, read_dataset, run


DATASET = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class EstimatorTests(unittest.TestCase):
    def test_dataset_parses_numeric_and_categorical_columns(self):
        x, y, numeric, categorical = read_dataset(DATASET)
        self.assertEqual(len(x), len(y))
        self.assertGreater(len(numeric), 0)
        self.assertGreater(len(categorical), 0)
        self.assertEqual(len(numeric) + len(categorical), len(x[0]))

    def test_model_predicts_finite_values(self):
        x, y, numeric, categorical = read_dataset(DATASET)
        model = build_model(numeric, categorical).fit(x[:70], y[:70])
        prediction = model.predict(x[70:])
        self.assertEqual(len(prediction), len(x) - 70)
        self.assertTrue(np.isfinite(prediction).all())

    def test_comparison_contains_both_estimators(self):
        result = run(DATASET)
        self.assertIn("model", result)
        self.assertIn("historical_baseline", result)
        self.assertGreater(result["test_rows"], 0)
        self.assertTrue(np.isfinite(result["model"]["mae"]))


if __name__ == "__main__":
    unittest.main()
