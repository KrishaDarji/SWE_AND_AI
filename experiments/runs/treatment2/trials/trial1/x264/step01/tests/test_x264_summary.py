import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import summarize_csv


class SummaryTests(unittest.TestCase):
    def test_classifies_settings_and_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("flag,quality\n0,2\n1,4\n", encoding="utf-8")
            result = summarize_csv(path)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["encoder_settings"][0]["name"], "flag")
        measure = result["quality_measures"][0]
        self.assertEqual(measure["name"], "quality")
        self.assertEqual(measure["mean"], 3.0)

    def test_real_dataset_shape(self):
        path = Path(__file__).parents[7] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = summarize_csv(path)
        self.assertEqual(result["rows"], 1152)
        self.assertEqual(len(result["encoder_settings"]), 16)
        self.assertEqual([item["name"] for item in result["quality_measures"]], ["Performance-"])



if __name__ == "__main__":
    unittest.main()
