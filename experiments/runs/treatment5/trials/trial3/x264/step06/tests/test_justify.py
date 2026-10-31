import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import justify

HEADER = "a,b,Performance-\n"
def fixture(text):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    handle.write(text); handle.close()
    return handle.name

class JustifyTests(unittest.TestCase):
    def test_selection_tie_reference_and_complete_config(self):
        path = fixture(HEADER + "1,2,10\n3,4,5\n9,8,5\n")
        try:
            result = justify.analyze(path)
            self.assertEqual(result["record_number"], 2)
            self.assertEqual(result["configuration"], {"a":"3", "b":"4", "Performance-":"5"})
            self.assertAlmostEqual(result["absolute_difference"], 5)
            self.assertAlmostEqual(result["percentage_improvement"], 50)
            rendered = justify.render(result, path)
            self.assertIn("Measured result", rendered); self.assertIn("Historical reference", rendered)
            self.assertIn("retrospective", rendered); self.assertIn("Associations", rendered)
        finally: os.unlink(path)

    def test_missing_nonfinite_and_zero_reference(self):
        path = fixture(HEADER + "1,2,NaN\n3,4,0\n5,6,inf\n")
        try:
            result = justify.analyze(path)
            self.assertEqual(result["usable"], 1)
            self.assertIsNone(result["percentage_improvement"])
            self.assertIn("undefined", justify.render(result, path))
        finally: os.unlink(path)

    def test_no_usable_records(self):
        path = fixture(HEADER + "1,2,missing\n3,4,nan\n")
        try:
            result = justify.analyze(path)
            self.assertEqual(result["usable"], 0)
            self.assertIn("No usable records", justify.render(result, path))
        finally: os.unlink(path)

if __name__ == "__main__": unittest.main()
