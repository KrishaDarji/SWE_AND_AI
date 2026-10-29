import csv
import tempfile
import unittest
from pathlib import Path

from source.analyze_x264 import DataError, identify_columns, load_dataset, summarize


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parents[5] / "moot/optimize/config/X264_AllMeasurements.csv"


class X264AnalyzerTests(unittest.TestCase):
    def test_load_and_identify_real_dataset(self):
        header, rows = load_dataset(DATASET)
        settings, performance = identify_columns(header, rows)
        self.assertEqual(len(rows), 1152)
        self.assertEqual(len(settings), 16)
        self.assertEqual(performance, "Performance-")

    def test_summary_contains_measured_sections(self):
        output = summarize(DATASET)
        self.assertIn("rows: 1152; columns: 17", output)
        self.assertIn("performance: min=", output)
        self.assertIn("strongest marginal Pearson relationships", output)
        self.assertIn("5-fold shuffled CV R2", output)

    def test_rejects_ragged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("a,Performance-\n1\n", encoding="utf-8")
            with self.assertRaises(DataError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
