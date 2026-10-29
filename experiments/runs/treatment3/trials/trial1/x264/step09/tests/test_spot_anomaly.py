import csv
import tempfile
import unittest
from pathlib import Path

from source.spot_anomaly import detect, load_csv


class SpotTests(unittest.TestCase):
    def test_real_dataset_is_numeric_and_has_expected_shape(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        names, X, y = load_csv(path)
        self.assertEqual(len(X), 1152)
        self.assertEqual(len(names), 16)
        self.assertEqual(len(y), len(X))

    def test_detector_returns_structured_result(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        result = detect(path)
        self.assertEqual(result["rows"], 1152)
        self.assertGreaterEqual(result["flagged_count"], 1)
        self.assertLessEqual(result["flagged_count"], result["rows"])
        self.assertTrue(all("row" in item for item in result["candidates"]))

    def test_bad_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "Performance-"])
                writer.writerow(["not-a-number", "1"])
            with self.assertRaises(ValueError):
                load_csv(path)


if __name__ == "__main__":
    unittest.main()
