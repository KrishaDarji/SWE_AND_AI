import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from selector import select


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"x": str(i)} for i in range(6)]
        self.values = {tuple(c.items()): float(i * 2) for i, c in enumerate(self.candidates)}

    def evaluator(self, candidate):
        return self.values[tuple(candidate.items())]

    def test_budget_and_membership(self):
        result = select(self.candidates, self.evaluator, 3)
        self.assertEqual(result["budget_used"], 3)
        self.assertIn(result["recommended_identifier"], result["evaluation_order"])

    def test_reproducibility(self):
        self.assertEqual(select(self.candidates, self.evaluator, 3), select(self.candidates, self.evaluator, 3))

    def test_duplicate_prevention(self):
        calls = []
        candidates = [{"x": "a"}, {"x": "a"}, {"x": "b"}]
        result = select(candidates, lambda c: calls.append(c) or 1.0, 20)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["budget_used"], 2)

    def test_hidden_outcomes_only_callback(self):
        hidden = {"a": 99.0, "b": 1.0}
        seen = []
        result = select([{"x": "a"}, {"x": "b"}], lambda c: seen.append(c) or hidden[c["x"]], 2)
        self.assertEqual(len(seen), 2)
        hidden["a"] = -1000.0
        self.assertNotEqual(result["recommended_identifier"], 0)

    def test_invalid_inputs(self):
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): select(self.candidates, self.evaluator, budget)
        with self.assertRaises(ValueError): select([], self.evaluator, 1)
        with self.assertRaises(ValueError): select(self.candidates, lambda c: math.nan, 1)
        with self.assertRaises(ValueError): select(self.candidates, lambda c: math.inf, 1)

    def test_budget_exceeds_candidates(self):
        result = select(self.candidates[:2], self.evaluator, 20)
        self.assertEqual(result["budget_used"], 2)


if __name__ == "__main__":
    unittest.main()
