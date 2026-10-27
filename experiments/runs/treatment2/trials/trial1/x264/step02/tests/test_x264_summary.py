import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import predict_encoding_time, summarize_csv


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
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = summarize_csv(path)
        self.assertEqual(result["rows"], 1152)
        self.assertEqual(len(result["encoder_settings"]), 16)
        self.assertEqual([item["name"] for item in result["quality_measures"]], ["Performance-"])

    def test_prediction_exact_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,20\n", encoding="utf-8")
            result = predict_encoding_time(path, {"a": "1", "b": "0"})
        self.assertEqual(result["method"], "exact-match")
        self.assertEqual(result["predicted_time"], 20.0)

    def test_prediction_unseen_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,1,30\n", encoding="utf-8")
            result = predict_encoding_time(path, {"a": "1", "b": "0"})
        self.assertEqual(result["method"], "inverse-hamming-weighted")
        self.assertEqual(result["nearest_distance"], 1)
        self.assertGreater(result["predicted_time"], 10)



if __name__ == "__main__":
    unittest.main()
