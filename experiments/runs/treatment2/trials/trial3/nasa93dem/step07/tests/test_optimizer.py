import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.optimizer import maximize_delivered_value


class OptimizerTests(unittest.TestCase):
    def test_maximizes_value_under_fixed_budget(self):
        history = [Project("h", 2000, 10.0, 10.0, None, None, {})]
        plan = maximize_delivered_value(history, [
            {"id": "a", "kloc": 5, "value": 8},
            {"id": "b", "kloc": 6, "value": 9},
            {"id": "c", "kloc": 4, "value": 7},
        ], 10)
        self.assertEqual(plan.selected_ids, ("b", "c"))
        self.assertEqual(plan.total_value, 16.0)
        self.assertEqual(plan.total_effort, 10.0)

    def test_rejects_unknown_or_negative_value(self):
        history = [Project("h", 2000, None, 10.0, None, None, {})]
        plan = maximize_delivered_value(history, [{"id": "x", "kloc": 5, "value": 2}], 10)
        self.assertEqual(plan.selected_ids, ())
        self.assertEqual(plan.rejected_ids, ("x",))

    def test_budget_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            maximize_delivered_value([], [], -1)


if __name__ == "__main__":
    unittest.main()
