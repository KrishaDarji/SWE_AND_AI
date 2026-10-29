import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarizer import read_rows, summarize


class SummarizerTests(unittest.TestCase):
    def test_real_dataset_shape_and_measure(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        fields, rows = read_rows(path)
        self.assertEqual(len(rows), 1152)
        self.assertIn("Performance-", fields)
        result = summarize(path)
        self.assertEqual(result["quality_measure"]["name"], "Performance-")
        self.assertEqual(len(result["setting_columns"]), 16)

    def test_summary_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("setting,Performance-\n")
            with self.assertRaises(ValueError):
                summarize(path)

    def test_constant_setting_is_reported(self):
        path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")
        result = summarize(path)
        self.assertEqual(result["settings"]["no_asm"]["unique"], 1)


if __name__ == "__main__":
    unittest.main()
