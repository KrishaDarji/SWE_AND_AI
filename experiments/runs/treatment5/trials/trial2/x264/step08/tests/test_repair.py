import importlib.util
import math
import unittest

SPEC = importlib.util.spec_from_file_location("repair", "source/repair.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FIELDS = ["a", "b"]


def row(order, a, b, outcome):
    return {"_order": order, "a": a, "b": b, "Performance-": outcome}


class RepairTests(unittest.TestCase):
    def test_minimum_change_beats_maximum_improvement(self):
        result = MODULE.repair(row(1, "1", "2", "10"), [row(1, 1, 2, 10), row(2, 1, 9, 9), row(3, 8, 7, 1)], FIELDS)
        self.assertEqual(result["repaired_identifier"], "record-2")
        self.assertEqual(result["change_count"], 1)

    def test_numeric_equivalence_and_exact_changes(self):
        result = MODULE.repair(row(1, "1.0", "2", "10"), [row(1, 1, 2, 10), row(2, 1.00, 3, 9)], FIELDS)
        self.assertEqual(result["changed_settings"], [{"setting": "b", "old": "2", "new": 3}])

    def test_tie_breaking(self):
        result = MODULE.repair(row(1, 1, 1, 10), [row(1, 1, 1, 10), row(2, 2, 1, 8), row(3, 1, 2, 8)], FIELDS)
        self.assertEqual(result["repaired_identifier"], "record-2")

    def test_no_improvement(self):
        result = MODULE.repair(row(1, 1, 1, 10), [row(1, 1, 1, 10), row(2, 2, 1, 10)], FIELDS)
        self.assertFalse(result["repaired"])

    def test_nonone_baseline_identifier_success(self):
        baseline = row(7, 1, 1, 10)
        result = MODULE.repair(baseline, [row(7, 1, 1, 10), row(8, 2, 1, 9)], FIELDS)
        self.assertEqual(result["baseline_identifier"], "record-7")

    def test_nonone_baseline_identifier_no_improvement(self):
        baseline = row(7, 1, 1, 10)
        result = MODULE.repair(baseline, [row(7, 1, 1, 10), row(8, 2, 1, 10)], FIELDS)
        self.assertFalse(result["repaired"])
        self.assertEqual(result["baseline_identifier"], "record-7")

    def test_invalid_baseline_and_missing_nonfinite(self):
        with self.assertRaises(ValueError):
            MODULE.repair({"a": 1, "b": 1, "Performance-": 10}, [], FIELDS)
        with self.assertRaises(ValueError):
            MODULE.repair({"a": 1, "Performance-": 10}, [], FIELDS)
        with self.assertRaises(ValueError):
            MODULE.repair(row(1, 1, 1, math.nan), [], FIELDS)
        with self.assertRaises(ValueError):
            MODULE.repair(row(1, 1, 1, 10), [row(1, 1, 1, "inf")], FIELDS)


if __name__ == "__main__":
    unittest.main()
