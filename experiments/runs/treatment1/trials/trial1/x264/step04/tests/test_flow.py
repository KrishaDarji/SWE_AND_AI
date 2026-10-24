import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow import FlowEvaluator


class FlowTests(unittest.TestCase):
    def test_streaming_ranking_and_recommendation(self):
        evaluator = FlowEvaluator(min_results=4, margin=0)
        self.assertFalse(evaluator.add({"ref": 1}, 100)["sufficient"])
        evaluator.add({"ref": 2}, 120)
        evaluator.add({"ref": 1}, 101)
        result = evaluator.add({"ref": 2}, 121)
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["recommended"]["configuration"], {"ref": 1})
        self.assertEqual(result["recommended"]["observations"], 2)

    def test_invalid_values(self):
        evaluator = FlowEvaluator()
        with self.assertRaises(ValueError):
            evaluator.add({}, 1)
        with self.assertRaises(ValueError):
            evaluator.add({"x": 1}, "bad")

    def test_one_observation_waits_for_noise_estimate(self):
        evaluator = FlowEvaluator(min_results=2)
        result = evaluator.add({"x": 1}, 1)
        self.assertEqual(result["reason"], "need more benchmark results")
        result = evaluator.add({"x": 2}, 2)
        self.assertFalse(result["sufficient"])
        self.assertIn("repeated", result["reason"])


if __name__ == "__main__":
    unittest.main()
