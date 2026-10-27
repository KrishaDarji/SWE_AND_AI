import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import load_dataset, predict, summarize


class SummaryAndPredictionTests(unittest.TestCase):
    def test_step1_real_dataset_summary_still_works(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        result = summarize(load_dataset(path))
        self.assertEqual(result["run_count"], 1152)
        self.assertEqual(result["setting_count"], 16)
        self.assertEqual(result["measures"], ["Performance-"])

    def test_step1_reusable_schema_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["fast", "quality", "latency"])
                writer.writerow([0, 10, 5])
                writer.writerow([1, 20, 3])
            result = summarize(load_dataset(path))
            self.assertEqual(result["settings"], ["fast"])
            self.assertEqual(result["quality"]["quality"]["mean"], 15.0)

    def test_prediction_exact_match(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        dataset = load_dataset(path)
        config = {name: int(float(dataset.rows[0][name])) for name in dataset.settings}
        result = predict(dataset, config)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["distance"], 0)
        self.assertGreater(result["neighbor_count"], 0)

    def test_prediction_unseen_configuration_uses_nearest_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,encoding_time\n0,0,10\n0,1,20\n1,0,30\n", encoding="utf-8")
            result = predict(load_dataset(path), {"a": 1, "b": 1})
            self.assertFalse(result["exact_match"])
            self.assertEqual(result["distance"], 1)
            self.assertEqual(result["prediction"], 25.0)


if __name__ == "__main__":
    unittest.main()
