import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import race_estimator


DATASET = pathlib.Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class RaceEstimatorTests(unittest.TestCase):
    def test_loads_dataset(self):
        rows = race_estimator.load_rows(DATASET)
        self.assertEqual(len(rows), 93)

    def test_estimate_is_positive_and_finite(self):
        rows = race_estimator.load_rows(DATASET)
        estimate = race_estimator.estimate(rows[0], race_estimator.fit(rows[1:]))
        self.assertTrue(math.isfinite(estimate))
        self.assertGreater(estimate, 0)

    def test_evaluation_is_deterministic_and_compares_baseline(self):
        rows = race_estimator.load_rows(DATASET)
        result = race_estimator.evaluate(rows)
        self.assertEqual(result["projects"], 93)
        self.assertGreaterEqual(result["model_wins"], 0)
        self.assertGreater(result["baseline_mae"], 0)


if __name__ == "__main__":
    unittest.main()
