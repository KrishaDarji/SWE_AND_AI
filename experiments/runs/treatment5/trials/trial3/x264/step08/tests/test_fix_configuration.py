import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from fix_configuration import repair


NAMES = ["a", "b"]


def row(i, a, b, outcome):
    return {"record_id": i, "settings": {"a": a, "b": b}, "outcome": outcome}


class RepairTests(unittest.TestCase):
    def test_minimum_change_beats_maximum_improvement(self):
        result = repair(row(1, "1", "2", "10"),
                        [row(1, "1", "2", "10"), row(2, "1", "3", "9"), row(3, "4", "5", "1")], NAMES)
        self.assertEqual(result["repaired_record"]["record_id"], 2)
        self.assertEqual(result["change_count"], 1)

    def test_numeric_equivalence_and_exact_changes(self):
        result = repair(row(1, "1.0", "2", "10"), [row(2, "1", "3.0", "9")], NAMES)
        self.assertEqual(result["changed_settings"], [{"setting": "b", "old": "2", "new": "3.0"}])

    def test_tie_breaks_by_outcome_then_record_order(self):
        result = repair(row(1, "1", "1", "10"), [row(2, "2", "1", "8"), row(3, "1", "2", "8")], NAMES)
        self.assertEqual(result["repaired_record"]["record_id"], 2)

    def test_no_improvement(self):
        self.assertFalse(repair(row(1, "1", "1", "10"), [row(2, "2", "2", "10")], NAMES)["repaired"])

    def test_invalid_baseline_and_missing_nonfinite_candidates(self):
        with self.assertRaises(ValueError):
            repair(row(1, "bad", "1", "10"), [], NAMES)
        result = repair(row(1, "1", "1", "10"), [row(2, "x", "1", "1"), row(3, "1", "1", "nan")], NAMES)
        self.assertFalse(result["repaired"])


if __name__ == "__main__":
    unittest.main()
