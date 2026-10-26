import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from flow import FlowEvaluator, ValidationResult


def result(name, mre, acc, pred):
    return ValidationResult({"name": name}, {"MRE-": mre, "ACC+": acc, "PRED40+": pred})


class FlowTests(unittest.TestCase):
    def test_updates_incrementally_and_uses_directions(self):
        evaluator = FlowEvaluator({"MRE-": "min", "ACC+": "max", "PRED40+": "max"})
        self.assertTrue(evaluator.observe(result("first", 0.4, 0.7, 0.5)))
        self.assertEqual(evaluator.best.configuration["name"], "first")
        self.assertFalse(evaluator.observe(result("worse", 0.5, 0.9, 0.9)))
        self.assertTrue(evaluator.observe(result("better", 0.3, 0.8, 0.6)))
        self.assertEqual(evaluator.best.configuration["name"], "better")

    def test_invalid_result_is_skipped(self):
        evaluator = FlowEvaluator({"score": "max"})
        self.assertFalse(evaluator.observe(ValidationResult({}, {"score": "bad"})))
        self.assertEqual(evaluator.invalid, 1)
        self.assertIsNone(evaluator.best)

    def test_equal_scores_keep_first_result(self):
        evaluator = FlowEvaluator({"score": "max"})
        self.assertTrue(evaluator.observe(ValidationResult({"name": "a"}, {"score": 1})))
        self.assertFalse(evaluator.observe(ValidationResult({"name": "b"}, {"score": 1})))
        self.assertEqual(evaluator.best.configuration["name"], "a")


if __name__ == "__main__":
    unittest.main()
