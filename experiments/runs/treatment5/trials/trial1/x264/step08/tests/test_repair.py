import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from repair import repair

S = ["a", "b"]
def row(a, b, score): return {"a": str(a), "b": str(b), "Performance-": str(score)}

class RepairTests(unittest.TestCase):
    def test_minimum_change_beats_maximum_improvement(self):
        base = row(0, 0, 10); records = [base, row(1, 0, 9), row(1, 1, 1)]
        answer = repair(base, records, S)
        self.assertEqual(answer["record_id"], 2); self.assertEqual(answer["change_count"], 1)
    def test_numeric_equivalence_and_exact_changes(self):
        base = row("1.0", "2", 10); answer = repair(base, [row(1, 2.0, 9)], S)
        self.assertEqual(answer["changed"], []); self.assertEqual(answer["change_count"], 0)
    def test_tie_order(self):
        answer = repair(row(0, 0, 10), [row(1, 0, 8), row(0, 1, 8)], S)
        self.assertEqual(answer["record_id"], 1)
    def test_no_improvement(self):
        self.assertFalse(repair(row(0, 0, 1), [row(0, 0, 1), row(1, 0, 2)], S)["repaired"])
    def test_invalid_missing_nonfinite(self):
        with self.assertRaises(ValueError): repair({"a": 0, "Performance-": 1}, [], S)
        with self.assertRaises(ValueError): repair(row(0, 0, math.nan), [], S)
        with self.assertRaises(ValueError): repair(row(0, 0, 1), [row(0, 0, "inf")], S)

if __name__ == "__main__": unittest.main()
