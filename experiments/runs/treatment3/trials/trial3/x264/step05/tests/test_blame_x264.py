import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("blame_x264", HERE / "source" / "blame_x264.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA = Path(__file__).resolve().parents[8] / "moot" / "optimize" / "config" / "X264_AllMeasurements.csv"

class BlameTests(unittest.TestCase):
    def test_load_and_target(self):
        fields, features, target, rows = mod.load_dataset(DATA)
        self.assertEqual(target, "Performance-")
        self.assertGreater(len(rows), 100)
        self.assertEqual(len(features), 16)

    def test_analysis_is_ranked_and_reproducible(self):
        result = mod.analyze(DATA)
        self.assertEqual(result["slow_count"], (result["rows"] + 1) // 2)
        self.assertEqual(len(result["ranked_settings"]), result["settings"])
        self.assertTrue(all("rf_permutation_mean" in item for item in result["ranked_settings"]))
        self.assertGreater(result["cv_r2_mean"], 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
