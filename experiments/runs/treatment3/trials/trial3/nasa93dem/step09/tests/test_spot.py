import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("spot", ROOT / "source" / "spot.py")
spot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spot)


class SpotTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("moot/optimize/process/nasa93dem.csv")

    def test_load_shape_and_required_columns(self):
        rows = spot.load_dataset(self.path)
        self.assertEqual(len(rows), 93)
        self.assertTrue(set(spot.REQUIRED).issubset(rows[0]))

    def test_detection_is_reproducible_and_returns_records(self):
        first = spot.detect(self.path)
        second = spot.detect(self.path)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first["suspicious"]), 1)
        self.assertTrue(all("id" in item and "reasons" in item for item in first["suspicious"]))


if __name__ == "__main__":
    unittest.main()
