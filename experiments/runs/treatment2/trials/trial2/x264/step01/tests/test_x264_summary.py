import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import load_dataset, summarize


class SummaryTests(unittest.TestCase):
    def test_load_and_summarize_real_dataset(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        dataset = load_dataset(path)
        result = summarize(dataset)
        self.assertEqual(result["run_count"], 1152)
        self.assertEqual(result["setting_count"], 16)
        self.assertEqual(result["measures"], ["Performance-"])
        self.assertEqual(result["quality"]["Performance-"]["count"], 1152)

    def test_reusable_schema_with_multiple_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["fast", "quality", "latency"])
                writer.writerow([0, 10, 5])
                writer.writerow([1, 20, 3])
            result = summarize(load_dataset(path))
            self.assertEqual(result["settings"], ["fast"])
            self.assertEqual(result["measures"], ["quality", "latency"])
            self.assertEqual(result["settings_by_value"]["fast"]["1"]["measures"]["quality"]["mean"], 20.0)


if __name__ == "__main__":
    unittest.main()
