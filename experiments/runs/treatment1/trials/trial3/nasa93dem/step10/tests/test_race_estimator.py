import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import race_estimator


class RaceEstimatorTests(unittest.TestCase):
    def test_loads_all_records_and_numeric_target(self):
        path = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        rows = race_estimator.load_rows(path)
        self.assertEqual(len(rows), 93)
        self.assertGreater(float(rows[0][race_estimator.TARGET]), 0)

    def test_knn_prediction_is_finite(self):
        rows = [{"prec": "h", **{name: "n" for name in race_estimator.FEATURES if name != "prec"}, "Effort-": "10"},
                {"prec": "l", **{name: "n" for name in race_estimator.FEATURES if name != "prec"}, "Effort-": "20"}]
        prediction = race_estimator.predict_knn(rows, rows[0], k=1)
        self.assertEqual(prediction, 10.0)

    def test_shared_evaluation_returns_same_metric_type(self):
        path = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        result = race_estimator.run(path)
        self.assertEqual(result["test_rows"], 31)
        self.assertGreaterEqual(result["estimator_mae"], 0.0)
        self.assertGreaterEqual(result["baseline_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
