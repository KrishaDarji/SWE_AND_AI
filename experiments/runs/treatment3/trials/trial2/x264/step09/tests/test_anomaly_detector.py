import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from anomaly_detector import detect, load_dataset


DATASET = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"


class DetectorTests(unittest.TestCase):
    def test_dataset_validates_and_is_reproducible(self):
        fields, features, records = load_dataset(DATASET)
        self.assertEqual(len(records), 1152)
        self.assertEqual(fields[-1], "Performance-")
        self.assertEqual(len(features), 16)
        self.assertEqual(detect(DATASET), detect(DATASET))

    def test_result_has_anomaly_and_configuration(self):
        result = detect(DATASET)
        self.assertGreaterEqual(result["flagged_count"], 1)
        self.assertEqual(len(result["top_configuration"]), 17)
        self.assertGreater(result["top_combined_score"], 0.0)

    def test_missing_target_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="") as handle:
                csv.writer(handle).writerow(["x"])
                csv.writer(handle).writerow(["1"])
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
