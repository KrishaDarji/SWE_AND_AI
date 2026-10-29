import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source" / "anomaly_detector.py"


class DetectorTests(unittest.TestCase):
    def test_real_dataset_validates_and_is_reproducible(self):
        first = subprocess.check_output([sys.executable, str(SOURCE)], text=True)
        second = subprocess.check_output([sys.executable, str(SOURCE)], text=True)
        a, b = json.loads(first), json.loads(second)
        self.assertEqual(a, b)
        self.assertEqual(a["rows"], 1152)
        self.assertEqual(a["features"], 16)
        self.assertEqual(len(a["suspicious_records"]), 10)

    def test_rejects_non_binary_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with open(path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["f", "Performance-"])
                writer.writerow(["2", "1.0"])
            proc = subprocess.run([sys.executable, str(SOURCE), str(path)], text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
