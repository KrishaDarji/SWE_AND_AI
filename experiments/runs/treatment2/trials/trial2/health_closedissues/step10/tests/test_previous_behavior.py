import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import (SequentialEvaluator, certify_families, predict_quality,
                           summarize_rows)


class AllBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.2", "ACC+": "0.90", "depth": "3", "criterion": "good"},
            {"MRE-": "0.3", "ACC+": "0.85", "depth": "4", "criterion": "good"},
            {"MRE-": "0.7", "ACC+": "0.40", "depth": "8", "criterion": "bad"},
            {"MRE-": "0.8", "ACC+": "0.45", "depth": "9", "criterion": "bad"},
        ]

    def test_summary_and_prediction_compatibility(self):
        summary = summarize_rows(self.rows, ["MRE-", "ACC+"], ["depth", "criterion"])
        self.assertEqual(summary["row_count"], 4)
        self.assertEqual(summary["configuration"]["criterion"]["values"], {"bad": 2, "good": 2})
        prediction = predict_quality(self.rows, {"depth": "3", "criterion": "good"},
                                     configuration_columns=["depth", "criterion"])
        self.assertAlmostEqual(prediction["predictions"]["ACC+"], .90, places=5)

    def test_certification_compatibility(self):
        result = certify_families(self.rows, "criterion", "good", "bad",
                                  metric="ACC+", resamples=500, seed=7)
        self.assertTrue(result["reliably_better"])

    def test_incremental_best_and_arrival_order(self):
        evaluator = SequentialEvaluator("ACC+")
        first = evaluator.add_result({"depth": 3}, {"ACC+": "0.70"})
        self.assertEqual(first["evaluated"], 1)
        evaluator.add_result({"depth": 5}, {"ACC+": 0.90})
        evaluator.add_result({"depth": 7}, {"ACC+": 0.80})
        result = evaluator.summary()
        self.assertEqual(result["evaluated"], 3)
        self.assertEqual(result["best"]["configuration"], {"depth": 5})
        self.assertEqual([r["configuration"]["depth"] for r in result["results"]], [3, 5, 7])

    def test_lower_is_better_and_ties_keep_first(self):
        evaluator = SequentialEvaluator("MRE-", higher_is_better=False)
        evaluator.add_result({"id": "first"}, {"MRE-": .2})
        evaluator.add_result({"id": "tie"}, {"MRE-": .2})
        self.assertEqual(evaluator.snapshot()["best"]["configuration"], {"id": "first"})

    def test_invalid_incremental_result_rejected(self):
        evaluator = SequentialEvaluator()
        with self.assertRaises(ValueError):
            evaluator.add_result({}, {"ACC+": .5})
        with self.assertRaises(ValueError):
            evaluator.add_result({"depth": 2}, {"MRE-": "not-a-number"})


if __name__ == "__main__":
    unittest.main()
