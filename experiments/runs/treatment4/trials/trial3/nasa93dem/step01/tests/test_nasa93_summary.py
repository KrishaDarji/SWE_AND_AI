import csv
import math
import os
import tempfile
import unittest

from source.nasa93_summary import summarize


class SummaryTests(unittest.TestCase):
    def make(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write(text); handle.close(); self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_all_required_behaviors(self):
        result = summarize(self.make("idX,cat,Kloc+,Effort-,Defects-,vX\n1,a,2,5,1,\n2,a,4,3,2,NaN\n3,b,,inf,3,7\n"))
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["missing_values"]["Effort-"], 1)
        self.assertEqual(result["missing_values"]["Kloc+"], 1)
        self.assertAlmostEqual(result["numeric_statistics"]["Kloc+"]["mean"], 3.0)
        self.assertEqual(result["categorical_frequencies"]["cat"], {"a": 2, "b": 1})
        self.assertEqual(result["roles"]["metadata_excluded_from_modeling"], ["idX", "vX"])
        self.assertEqual(result["roles"]["outcomes"], ["Kloc+", "Effort-", "Defects-"])
        self.assertEqual(result["outcome_directions"], {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"})

    def test_empty_input(self):
        result = summarize(self.make(""))
        self.assertEqual(result["record_count"], 0)
        self.assertEqual(result["columns"], [])
        self.assertEqual(result["numeric_statistics"], {})


if __name__ == "__main__":
    unittest.main()
