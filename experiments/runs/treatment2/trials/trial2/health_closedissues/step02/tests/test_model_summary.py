import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import predict_quality, summarize_rows


class SummaryAndPredictionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.2", "ACC+": "0.8", "depth": "3", "criterion": "a"},
            {"MRE-": "0.6", "ACC+": "0.4", "depth": "9", "criterion": "b"},
        ]

    def test_existing_summary_behavior(self):
        result = summarize_rows(self.rows, ["MRE-", "ACC+"], ["depth", "criterion"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["quality_metrics"]["MRE-"]["median"], 0.4)
        self.assertEqual(result["configuration"]["criterion"]["values"], {"a": 1, "b": 1})

    def test_prediction_uses_similar_configuration(self):
        result = predict_quality(self.rows, {"depth": "3", "criterion": "a"},
                                 configuration_columns=["depth", "criterion"])
        self.assertAlmostEqual(result["predictions"]["MRE-"], 0.2, places=6)
        self.assertAlmostEqual(result["predictions"]["ACC+"], 0.8, places=6)
        self.assertEqual(result["neighbors_used"], 2)

    def test_invalid_prediction_target_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_quality(self.rows, {"depth": "3"}, configuration_columns=["depth", "criterion"])


if __name__ == "__main__":
    unittest.main()
