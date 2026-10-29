import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify_x264 import load, bootstrap_mean

DATA = Path(__file__).parents[8] / "moot" / "optimize" / "config" / "X264_AllMeasurements.csv"

class JustifyTests(unittest.TestCase):
    def test_dataset_validates_and_is_binary(self):
        features, X, y, rows = load(DATA)
        self.assertEqual((len(features), len(X), len(y)), (16, 1152, 1152))
        self.assertEqual(max(y), 821.963)
        self.assertTrue(all(v in (0.0, 1.0) for row in X for v in row))

    def test_bootstrap_is_reproducible_and_ordered(self):
        lo, hi = bootstrap_mean([1,2,3], n=1000)
        self.assertLessEqual(lo, hi)
        self.assertEqual((lo, hi), bootstrap_mean([1,2,3], n=1000))

    def test_invalid_nonbinary_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,Performance-\n2,1\n"); name=f.name
        with self.assertRaises(ValueError): load(name)

if __name__ == "__main__": unittest.main()
