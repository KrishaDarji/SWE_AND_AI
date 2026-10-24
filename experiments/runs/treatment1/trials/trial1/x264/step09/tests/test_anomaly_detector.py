import csv
import pathlib
import tempfile
import unittest

from source.anomaly_detector import detect


class DetectorTests(unittest.TestCase):
    def write(self, values):
        handle = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(handle)
        writer.writerow(["feature", "Performance-"])
        for i, value in enumerate(values):
            writer.writerow([i, value])
        handle.close()
        self.addCleanup(pathlib.Path(handle.name).unlink)
        return handle.name

    def test_flags_robust_extreme(self):
        report = detect(self.write([10, 11, 10, 9, 10, 1000]))
        self.assertEqual([item["row"] for item in report["outliers"]], [7])

    def test_does_not_flag_normal_spread(self):
        self.assertEqual(detect(self.write([10, 11, 12, 13, 14]))["outliers"], [])

    def test_invalid_measurement_is_reported(self):
        report = detect(self.write(["bad", 10, 11]))
        self.assertEqual(report["invalid_rows"], [2])


if __name__ == "__main__":
    unittest.main()
