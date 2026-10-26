import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

SOURCE = os.path.join(os.path.dirname(__file__), "..", "source")
sys.path.insert(0, SOURCE)
from predictor import EncodingTimePredictor, load_measurements


class PredictorTests(unittest.TestCase):
    def test_learns_linear_relationship(self):
        model = EncodingTimePredictor(["a", "b"], [([0, 0], 2), ([1, 0], 4), ([0, 1], 5)])
        self.assertAlmostEqual(model.predict({"a": 1, "b": 1}), 7, places=4)

    def test_rejects_missing_and_extra_fields(self):
        model = EncodingTimePredictor(["a"], [([0], 3)])
        with self.assertRaises(ValueError):
            model.predict({})
        with self.assertRaises(ValueError):
            model.predict({"a": 0, "b": 1})

    def test_csv_loader_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = os.path.join(directory, "data.csv")
            config = os.path.join(directory, "config.json")
            with open(dataset, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["a", "Performance-"])
                writer.writerow(["0", "2"])
                writer.writerow(["1", "4"])
            with open(config, "w") as handle:
                json.dump({"a": 0.5}, handle)
            self.assertEqual(load_measurements(dataset)[0], ["a"])
            result = subprocess.run([sys.executable, os.path.join(SOURCE, "predictor.py"),
                                     "--dataset", dataset, "--config", config],
                                    capture_output=True, text=True, check=True)
            self.assertIn("predicted_encoding_time", result.stdout)


if __name__ == "__main__":
    unittest.main()
