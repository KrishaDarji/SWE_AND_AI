import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from selector import choose, read_dataset

DATA = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"

class SelectorTests(unittest.TestCase):
    def test_dataset_shape_and_types(self):
        rows, features, target = read_dataset(DATA)
        self.assertEqual(len(rows), 1152)
        self.assertEqual(len(features), 16)
        self.assertEqual(target, "Performance-")

    def test_quality_constraint_is_applied(self):
        result = choose(DATA)
        self.assertGreater(result["feasible_count"], 0)
        self.assertLess(result["feasible_count"], result["total_count"])

    def test_empty_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.csv"
            p.write_text("a,Performance-\n")
            with self.assertRaises(ValueError):
                choose(p)

if __name__ == "__main__":
    unittest.main()
