import math
import unittest
from source.choose import select

class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [(0,), (1,), (2,), (3,), (3,)]
        self.values = {(0,): 4.0, (1,): 3.0, (2,): 1.0, (3,): 2.0}

    def test_budget_and_duplicate_prevention(self):
        calls = []
        result = select(self.candidates, lambda c: calls.append(c) or self.values[c], 3)
        self.assertEqual(result.budget_used, 3)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertLessEqual(len(calls), 3)

    def test_reproducible_and_membership(self):
        f = lambda c: self.values[c]
        a, b = select(self.candidates, f, 3), select(self.candidates, f, 3)
        self.assertEqual(a, b)
        self.assertIn(a.recommended_id, a.evaluation_order)
        self.assertIn(a.recommended_configuration, [x[1] for x in a.observations])

    def test_hidden_outcomes_only_callback(self):
        first = dict(self.values)
        second = dict(self.values)
        result_a = select(self.candidates, lambda c: first[c], 2)
        evaluated = {row[1] for row in result_a.observations}
        for config in set(second) - evaluated:
            second[config] = -100000.0
        result_b = select(self.candidates, lambda c: second[c], 2)
        self.assertEqual(result_a, result_b)

    def test_invalid_inputs(self):
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): select([(1,)], lambda c: 1, budget)
        with self.assertRaises(ValueError): select([], lambda c: 1, 1)
        with self.assertRaises(ValueError): select([(1,)], lambda c: math.nan, 1)
        result = select([(1,), (2,)], lambda c: 1, 99)
        self.assertEqual(result.budget_used, 2)

if __name__ == "__main__": unittest.main()
