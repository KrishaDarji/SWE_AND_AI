import csv
import math
import os
import tempfile
import unittest

from source.choose import DESCRIPTORS, choose

HEADER = ["idX"] + DESCRIPTORS + ["Kloc+", "Effort-", "Defects-", "MonthsX"]

def row(identifier, size, effort, **overrides):
    values = {name: "h" for name in DESCRIPTORS}
    values.update({"idX": identifier, "Kloc+": size, "Effort-": effort,
                   "Defects-": "1", "MonthsX": "1"})
    values.update(overrides)
    return [values[name] for name in HEADER]

class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "fixture.csv")

    def tearDown(self):
        self.directory.cleanup()

    def write(self, rows):
        with open(self.path, "w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(HEADER); writer.writerows(rows)

    def test_budget_boundary_and_maximum(self):
        self.write([row("a", 10, 0), row("b", 20, 100), row("c", 99, 100.1)])
        result = choose(self.path, 100)
        self.assertEqual(result["feasible"], 2); self.assertEqual(result["selected"]["idX"], "b")

    def test_ties_effort_then_earliest_position(self):
        self.write([row("first", 50, 9), row("low-effort", 50, 8), row("earliest", 50, 8)])
        result = choose(self.path, 10)
        self.assertEqual(result["selected"]["idX"], "low-effort")
        self.write([row("duplicate", 50, 8), row("duplicate", 50, 8)])
        result = choose(self.path, 10)
        self.assertEqual(result["selected"]["position"], 1)

    def test_invalid_budget(self):
        self.write([row("a", 1, 1)])
        for budget in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): choose(self.path, budget)

    def test_missing_and_nonfinite_outcomes_are_infeasible(self):
        self.write([row("bad-effort", 20, "nan"), row("bad-size", "", 1), row("ok", 3, 2)])
        result = choose(self.path, 2)
        self.assertEqual(result["feasible"], 1); self.assertEqual(result["selected"]["idX"], "ok")

    def test_empty_and_no_feasible(self):
        with open(self.path, "w") as stream: stream.write(",".join(HEADER) + "\n")
        with self.assertRaises(ValueError): choose(self.path, 1)
        self.write([row("too-costly", 100, 2)])
        result = choose(self.path, 1)
        self.assertIsNone(result["selected"]); self.assertEqual(result["feasible"], 0)

    def test_metadata_outcomes_separate_and_original_position_survives(self):
        self.write([row("bad", "nan", 1), row("same", 7, 1), row("same", 6, 1)])
        result = choose(self.path, 1)
        selected = result["selected"]
        self.assertEqual(selected["position"], 2); self.assertEqual(selected["idX"], "same")
        self.assertNotIn("Kloc+", selected["descriptors"]); self.assertNotIn("Effort-", selected["descriptors"])

if __name__ == "__main__": unittest.main()
