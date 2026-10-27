import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import certify_families, predict_quality, summarize_rows


class CompatibilityAndCertificationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.2", "ACC+": "0.90", "depth": "3", "criterion": "good"},
            {"MRE-": "0.3", "ACC+": "0.85", "depth": "4", "criterion": "good"},
            {"MRE-": "0.7", "ACC+": "0.40", "depth": "8", "criterion": "bad"},
            {"MRE-": "0.8", "ACC+": "0.45", "depth": "9", "criterion": "bad"},
        ]

    def test_step1_summary_behavior(self):
        result = summarize_rows(self.rows, ["MRE-", "ACC+"], ["depth", "criterion"])
        self.assertEqual(result["row_count"], 4)
        self.assertEqual(result["configuration"]["criterion"]["values"], {"bad": 2, "good": 2})

    def test_step2_prediction_behavior(self):
        result = predict_quality(self.rows, {"depth": "3", "criterion": "good"},
                                 configuration_columns=["depth", "criterion"])
        self.assertAlmostEqual(result["predictions"]["ACC+"], 0.90, places=5)
        self.assertEqual(result["neighbors_used"], 4)

    def test_certifies_reliably_better_family(self):
        result = certify_families(self.rows, "criterion", "good", "bad",
                                  metric="ACC+", resamples=5000, seed=7)
        self.assertTrue(result["reliably_better"])
        self.assertEqual(result["sample_sizes"], {"a": 2, "b": 2})

    def test_lower_metric_direction(self):
        result = certify_families(self.rows, "criterion", "good", "bad",
                                  metric="MRE-", higher_is_better=False,
                                  resamples=1000, seed=7)
        self.assertTrue(result["reliably_better"])

    def test_missing_family_rejected(self):
        with self.assertRaises(ValueError):
            certify_families(self.rows, "criterion", "missing", "bad")


if __name__ == "__main__":
    unittest.main()
