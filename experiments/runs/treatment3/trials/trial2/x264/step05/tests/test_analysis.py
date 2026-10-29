import csv
import tempfile
import unittest
from pathlib import Path

from source.analyze_x264 import TARGET, analyze, load_dataset


ROOT = Path(__file__).resolve().parents[8]
DATASET = ROOT / "moot/optimize/config/X264_AllMeasurements.csv"


class AnalysisTests(unittest.TestCase):
    def test_load_validates_shape_and_binary_settings(self):
        settings, X, y = load_dataset(DATASET)
        self.assertEqual(len(settings), 16)
        self.assertEqual(X.shape, (1152, 16))
        self.assertEqual(y.shape, (1152,))
        self.assertEqual(TARGET, "Performance-")

    def test_analysis_has_complete_ranking_and_slow_group(self):
        report = analyze(DATASET)
        self.assertEqual(report["rows"], 1152)
        self.assertEqual(len(report["ranking"]), 16)
        self.assertGreater(report["slow_rows"], 0)
        self.assertTrue(all("permutation_importance_mean" in item for item in report["ranking"]))

    def test_rejects_non_binary_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["setting", TARGET])
                writer.writerow([2, 1])
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
