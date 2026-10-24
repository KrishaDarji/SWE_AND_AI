import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import certify


class CertifyTests(unittest.TestCase):
    def test_clear_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["setting", "Performance-"])
                for setting, value in [(0, 10), (1, 20), (2, 21), (3, 22), (4, 23), (5, 24)]:
                    writer.writerow([setting, value])
            result = certify(path, neighbors=2)
            self.assertEqual(result["best_measurement"], 10.0)
            self.assertTrue(result["meaningfully_faster"])

    def test_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("setting,other\n1,2\n")
            with self.assertRaises(ValueError):
                certify(path)


if __name__ == "__main__":
    unittest.main()
