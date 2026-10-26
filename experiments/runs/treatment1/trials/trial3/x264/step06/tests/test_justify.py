import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify import read_measurements, recommend


class JustifyTests(unittest.TestCase):
    def make_csv(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_minimum_for_trailing_minus_metric(self):
        path = self.make_csv("a,b,Performance-\n0,1,9\n1,0,3\n")
        result = recommend(path)
        self.assertEqual(result["recommended_configuration"], {"a": "1", "b": "0"})
        self.assertEqual(result["rank"], 1)
        self.assertEqual(result["direction"], "minimize")

    def test_maximum_for_metric_without_minus(self):
        path = self.make_csv("a,quality\n0,2\n1,8\n")
        result = recommend(path)
        self.assertEqual(result["recommended_configuration"], {"a": "1"})
        self.assertEqual(result["direction"], "maximize")

    def test_real_dataset_shape(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        fields, target, rows = read_measurements(path)
        self.assertEqual(len(fields), 16)
        self.assertEqual(target, "Performance-")
        self.assertEqual(len(rows), 1152)


if __name__ == "__main__":
    unittest.main()
