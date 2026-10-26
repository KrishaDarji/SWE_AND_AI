import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import race_estimator


DATASET = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"


class RaceEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = race_estimator.load_rows(DATASET.resolve())

    def test_dataset_shape_and_numeric_targets(self):
        self.assertEqual(len(self.rows), 93)
        self.assertTrue(all(isinstance(row["Effort-"], float) for row in self.rows))

    def test_leave_one_out_predictions_are_finite(self):
        result = race_estimator.evaluate(self.rows)
        self.assertEqual(result["projects"], 93)
        self.assertTrue(math.isfinite(result["estimator_mae"]))
        self.assertTrue(math.isfinite(result["historical_baseline_mae"]))

    def test_baseline_uses_training_only(self):
        training = self.rows[1:]
        prediction = race_estimator.historical_prediction(training, self.rows[0])
        expected = race_estimator.statistics.median(
            row["Effort-"] / row["Kloc+"] for row in training if row["Kloc+"] > 0
        ) * self.rows[0]["Kloc+"]
        self.assertEqual(prediction, expected)


if __name__ == "__main__":
    unittest.main()
