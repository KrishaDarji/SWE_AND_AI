import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from choose import DESCRIPTORS, choose, report

def rec(identifier, effort, size, position, **fields):
    row = {name: "setting" for name in DESCRIPTORS}
    row.update({"idX": str(identifier), "Effort-": str(effort), "Kloc+": str(size)})
    row.update(fields)
    return {"position": position, "row": row}

class ChooseTests(unittest.TestCase):
    def test_boundary_and_maximum(self):
        result = choose([rec("a", 10, 20, 1), rec("b", 11, 99, 2)], 10)
        self.assertEqual(result["selected"]["row"]["idX"], "a")
        self.assertEqual(result["feasible"], 1)

    def test_tie_lower_effort(self):
        result = choose([rec("slow", 8, 50, 1), rec("fast", 4, 50, 2)], 8)
        self.assertEqual(result["selected"]["row"]["idX"], "fast")

    def test_tie_earliest_csv_record(self):
        result = choose([rec("later", 4, 50, 9), rec("earlier", 4, 50, 3)], 8)
        self.assertEqual(result["selected"]["row"]["idX"], "earlier")

    def test_invalid_budget(self):
        with self.assertRaises(ValueError): choose([], -1)
        with self.assertRaises(ValueError): choose([], "nan")
        with self.assertRaises(ValueError): choose([], "bad")

    def test_missing_and_nonfinite_outcomes(self):
        result = choose([rec("bad1", 2, "nan", 1), rec("bad2", "inf", 8, 2)], 10)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["feasible"], 0)

    def test_empty_and_no_feasible(self):
        self.assertIsNone(choose([], 0)["selected"])
        self.assertEqual(choose([rec("x", 2, 8, 1)], 1)["feasible"], 0)

    def test_metadata_separate_from_descriptors_and_duplicate_ids(self):
        result = choose([rec("dup", 2, 8, 1), rec("dup", 1, 7, 2, tool="special")], 2)
        text = report(result)
        self.assertIn("original_record_position=1", text)
        self.assertIn("descriptor_settings=", text)
        self.assertIn("observed_effort=2", text)
        self.assertNotIn("Effort-=2", text.split("descriptor_settings=", 1)[1])

    def test_original_position_survives_invalid_filtering(self):
        result = choose([rec("invalid", 1, "nan", 1), rec("kept", 3, 4, 2)], 3)
        self.assertEqual(result["selected"]["position"], 2)

if __name__ == "__main__":
    unittest.main()
