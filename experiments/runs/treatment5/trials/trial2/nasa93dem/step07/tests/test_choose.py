import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from choose import DESCRIPTORS, SelectionError, choose


def row(identifier, effort, size, **extra):
    value = {"idX": identifier, "Kloc+": size, "Effort-": effort}
    value.update({name: "setting" for name in DESCRIPTORS})
    value.update(extra)
    return value


class ChooseTests(unittest.TestCase):
    def test_budget_boundary_and_maximum(self):
        result = choose([row("a", 10, 5), row("b", 20, 9), row("c", 21, 99)], 20)
        self.assertEqual((result["record_id"], result["feasible_count"]), ("b", 2))

    def test_tie_lower_effort(self):
        self.assertEqual(choose([row("a", 8, 7), row("b", 3, 7)], 10)["record_id"], "b")

    def test_tie_earliest_position_and_duplicate_ids(self):
        result = choose([row("same", 3, 7), row("same", 3, 7)], 10)
        self.assertEqual(result["record_position"], 1)

    def test_invalid_budget(self):
        for budget in (-1, "nan", "inf"):
            with self.assertRaises(SelectionError):
                choose([row("a", 1, 1)], budget)

    def test_missing_and_nonfinite_outcomes(self):
        with self.assertRaises(SelectionError):
            choose([row("a", None, 1)], 2)
        with self.assertRaises(SelectionError):
            choose([row("a", 1, math.inf)], 2)

    def test_empty_and_no_feasible(self):
        with self.assertRaises(SelectionError):
            choose([], 2)
        with self.assertRaises(SelectionError):
            choose([row("a", 3, 1)], 2)

    def test_metadata_is_not_descriptor_settings(self):
        result = choose([row("id", 1, 4, centerX="metadata")], 1)
        self.assertNotIn("idX", result["descriptors"])
        self.assertNotIn("Kloc+", result["descriptors"])
        self.assertEqual(result["descriptors"]["prec"], "setting")


if __name__ == "__main__":
    unittest.main()
