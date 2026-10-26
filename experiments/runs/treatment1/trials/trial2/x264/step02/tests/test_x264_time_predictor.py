import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_time_predictor import load_dataset, predict


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".csv"))
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["a", "b", "Performance-"])
            writer.writerows([[0, 0, 10], [0, 1, 20], [1, 1, 30]])

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_load_and_exact_neighbor(self):
        features, records = load_dataset(self.path)
        result = predict(features, records, {"a": 0, "b": 1}, 1)
        self.assertEqual(result["predicted_seconds"], 20.0)
        self.assertEqual(result["nearest_distance"], 0)

    def test_weighted_neighbors_and_validation(self):
        features, records = load_dataset(self.path)
        result = predict(features, records, {"a": 0, "b": 0}, 3)
        self.assertAlmostEqual(result["predicted_seconds"], (10 + 20 / 2 + 30 / 3) / (1 + .5 + 1 / 3))
        with self.assertRaises(ValueError):
            predict(features, records, {"a": 0}, 1)
        with self.assertRaises(ValueError):
            predict(features, records, {"a": 2, "b": 0}, 1)


if __name__ == "__main__":
    unittest.main()
