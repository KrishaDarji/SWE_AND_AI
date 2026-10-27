import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import certify, load_dataset, predict, summarize


REAL = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"


class CompatibilityAndCertificationTests(unittest.TestCase):
    def test_previous_summary_and_prediction_behavior(self):
        dataset = load_dataset(REAL)
        self.assertEqual(summarize(dataset)["run_count"], 1152)
        config = {name: int(float(dataset.rows[0][name])) for name in dataset.settings}
        result = predict(dataset, config)
        self.assertTrue(result["exact_match"])
        self.assertGreater(result["neighbor_count"], 0)

    def test_previous_reusable_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("fast,quality,latency\n0,10,5\n1,20,3\n", encoding="utf-8")
            result = summarize(load_dataset(path))
            self.assertEqual(result["quality"]["quality"]["mean"], 15.0)

    def test_certification_detects_credible_improvement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "quality"])
                for value in (10, 11, 9, 10): writer.writerow([0, value])
                for value in (20, 21, 19, 20): writer.writerow([1, value])
            result = certify(load_dataset(path))
            self.assertTrue(result["any_credible"])
            self.assertIn("feature", result["credible_improvements"])
            self.assertGreater(result["settings"]["feature"]["improvement"], 0)

    def test_real_dataset_certification_is_reported(self):
        result = certify(load_dataset(REAL))
        self.assertEqual(result["measure"], "Performance-")
        self.assertEqual(len(result["settings"]), 16)
        self.assertEqual(result["settings"]["no_asm"]["status"], "not_testable")
        self.assertEqual(result["settings"]["no_8x8dct"]["status"], "tested")


if __name__ == "__main__":
    unittest.main()
