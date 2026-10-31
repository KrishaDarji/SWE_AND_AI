import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from selector import select_configurations


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.items = [(i,) for i in range(8)]
        self.outcomes = {item: 100.0 - item[0] for item in self.items}

    def test_budget_reproducibility_and_membership(self):
        seen = []
        def evaluate(config):
            seen.append(config)
            return self.outcomes[config]
        first = select_configurations(self.items, 3, evaluate)
        second = select_configurations(self.items, 3, lambda c: self.outcomes[c])
        self.assertEqual(first, second)
        self.assertEqual(len(seen), 3)
        self.assertIn(first.recommended_config, seen)

    def test_duplicate_prevention_and_excess_budget(self):
        calls = []
        items = [(1,), (1,), (2,)]
        result = select_configurations(items, 99, lambda c: calls.append(c) or c[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result.budget_used, 2)

    def test_hidden_outcomes_only_reached_by_callback(self):
        hidden = {item: value for item, value in zip(self.items, range(8))}
        accessed = []
        class Guard(dict):
            def __getitem__(self, key):
                accessed.append(key)
                return super().__getitem__(key)
        guarded = Guard(hidden)
        result = select_configurations(self.items, 4, lambda c: guarded[c])
        self.assertEqual(len(accessed), 4)
        changed = Guard({**hidden, (0,): -1000})
        other = select_configurations(self.items, 4, lambda c: changed[c])
        self.assertEqual(result.recommended_id, other.recommended_id)
        self.assertEqual(result.recommended_config, other.recommended_config)

    def test_invalid_inputs(self):
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                select_configurations(self.items, budget, lambda c: 1)
        with self.assertRaises(ValueError):
            select_configurations([], 1, lambda c: 1)
        with self.assertRaises(ValueError):
            select_configurations(self.items, 1, lambda c: math.inf)


if __name__ == "__main__":
    unittest.main()
