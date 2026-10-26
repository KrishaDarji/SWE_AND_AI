import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from hpo_fix import Config, change_count, format_result, suggest  # noqa: E402


class SuggestionTests(unittest.TestCase):
    def test_minimizes_changed_parameters_then_error(self):
        base = Config(0.5, {"N_estimators": 10, "criterion": "a", "Min_sample_leaves": 2, "Min_impurity_decrease": 1.0, "Max_depth": 5}, 2)
        one_change = Config(0.4, {**base.values, "Max_depth": 6}, 3)
        two_changes = Config(0.1, {**base.values, "Max_depth": 7, "N_estimators": 20}, 4)
        b, chosen, count = suggest([base, two_changes, one_change])
        self.assertIs(b, base)
        self.assertIs(chosen, one_change)
        self.assertEqual(count, 1)

    def test_tie_uses_lower_error(self):
        base = Config(0.5, {"N_estimators": 10, "criterion": "a", "Min_sample_leaves": 2, "Min_impurity_decrease": 1.0, "Max_depth": 5}, 2)
        high = Config(0.4, {**base.values, "Max_depth": 6}, 3)
        low = Config(0.3, {**base.values, "Max_depth": 7}, 4)
        self.assertIs(suggest([base, high, low])[1], low)

    def test_change_count_and_output(self):
        base = Config(1.0, {"N_estimators": 1, "criterion": "a", "Min_sample_leaves": 1, "Min_impurity_decrease": 0.0, "Max_depth": 1}, 2)
        chosen = Config(0.5, {**base.values, "criterion": "b"}, 3)
        self.assertEqual(change_count(base, chosen), 1)
        self.assertIn("criterion: a -> b", format_result(base, chosen, 1))


if __name__ == "__main__":
    unittest.main()
