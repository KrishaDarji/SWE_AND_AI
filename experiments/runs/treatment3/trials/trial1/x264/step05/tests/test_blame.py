import csv
import tempfile
import unittest
from pathlib import Path

from source.blame import analyze, load_data


class BlameTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkstemp(suffix=".csv")[1])
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fast", "slow", "Performance-"])
            writer.writerows([[1, 0, 10], [1, 0, 11], [0, 1, 20], [0, 1, 21], [1, 0, 10]])

    def tearDown(self):
        self.path.unlink()

    def test_load_and_blame(self):
        names, x, y = load_data(self.path)
        self.assertEqual(names, ["fast", "slow"])
        self.assertEqual(len(x), len(y))
        report = analyze(self.path)
        self.assertEqual(report["rows"], 5)
        self.assertEqual(report["features"], 2)
        self.assertGreater(report["results"][0]["enabled_minus_disabled"], 0)

    def test_empty_dataset_rejected(self):
        empty = self.path.with_name("empty.csv")
        empty.write_text("fast,Performance-\n")
        with self.assertRaises(ValueError):
            load_data(empty)
        empty.unlink()


if __name__ == "__main__":
    unittest.main()
