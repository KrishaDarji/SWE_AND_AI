import csv
import pathlib
import sys
import tempfile
import unittest

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
import x264_summary


class SummaryTests(unittest.TestCase):
    def test_classifies_binary_settings_and_measures(self):
        rows = [
            {"setting": "0", "quality": "2"},
            {"setting": "1", "quality": "4"},
            {"setting": "1", "quality": "3"},
        ]
        result = x264_summary.summarize(rows)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["settings"][0][0], "setting")
        self.assertEqual(result["measures"][0][0], "quality")
        self.assertIn("mean 3.000", x264_summary.format_summary(result))

    def test_reads_real_dataset(self):
        path = pathlib.Path(__file__).resolve().parents[7] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = x264_summary.summarize(x264_summary.read_csv(path))
        self.assertEqual(result["rows"], 1152)
        self.assertEqual(len(result["settings"]), 16)
        self.assertEqual([name for name, _ in result["measures"]], ["Performance-"])

    def test_cli_accepts_whitespace_and_empty_path_errors(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
            writer = csv.writer(handle)
            writer.writerow(["flag", "measure"])
            writer.writerow(["1", " 5.5 "])
            handle.flush()
            self.assertEqual(x264_summary.main([handle.name]), 0)
        self.assertEqual(x264_summary.main(["/does/not/exist.csv"]), 1)


if __name__ == "__main__":
    unittest.main()
