import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow import FlowEvaluator


class FlowTests(unittest.TestCase):
    def test_arrivals_update_cumulative_accuracy(self):
        evaluator = FlowEvaluator(tolerance=0.20)
        first = evaluator.add("a", 100, 100)
        self.assertEqual(first["observations"], 1)
        result = evaluator.add("b", 80, 100)
        self.assertEqual(result["mean_relative_error"], 0.1)
        self.assertEqual(result["pass_rate"], 1.0)

    def test_invalid_measurements(self):
        evaluator = FlowEvaluator()
        with self.assertRaises(ValueError):
            evaluator.add("a", -1, 2)
        with self.assertRaises(ValueError):
            evaluator.add("a", 1, 0)
        with self.assertRaises(ValueError):
            evaluator.add("a", "bad", 2)

    def test_empty_summary(self):
        self.assertEqual(FlowEvaluator().summary()["observations"], 0)


if __name__ == "__main__":
    unittest.main()
