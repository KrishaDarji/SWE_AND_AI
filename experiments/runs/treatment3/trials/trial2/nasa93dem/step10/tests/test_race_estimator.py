import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "source"))
from race_estimator import load_dataset, RaceEstimator, historical_baseline, metrics


class RaceTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = load_dataset(os.path.join(ROOT, "../../../../../../../moot/optimize/process/nasa93dem.csv"))
        self.assertEqual(len(rows), 93)
        self.assertTrue(all(float(r["Effort-"]) > 0 for r in rows))

    def test_predict_shape_and_finite(self):
        rows = load_dataset(os.path.join(ROOT, "../../../../../../../moot/optimize/process/nasa93dem.csv"))
        model = RaceEstimator().fit(rows[:70])
        pred = model.predict(rows[70:])
        self.assertEqual(len(pred), 23)
        self.assertTrue(all(p > 0 for p in pred))

    def test_metrics_and_baseline(self):
        rows = load_dataset(os.path.join(ROOT, "../../../../../../../moot/optimize/process/nasa93dem.csv"))
        actual = [float(r["Effort-"]) for r in rows[70:]]
        pred = historical_baseline(rows[:70], rows[70:])
        self.assertEqual(len(pred), len(actual))
        self.assertIn("rmse", metrics(actual, pred))


if __name__ == "__main__":
    unittest.main()
