import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import justify_config


class JustificationTests(unittest.TestCase):
    def test_load_and_recommend_real_dataset(self):
        path = "/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv"
        result = justify_config.recommend(path)
        self.assertGreater(result["rows"], 1)
        self.assertEqual(len(result["configuration"]), 16)
        self.assertLessEqual(result["observed_performance"], result["dataset_mean"])
        self.assertIn("Recommendation", justify_config.render(result))

    def test_rejects_missing_target(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a\n1\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                justify_config.load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
