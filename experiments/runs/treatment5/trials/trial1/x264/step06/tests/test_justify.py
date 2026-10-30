import csv
import math
import os
import tempfile
import unittest

from source.justify import analyze, render


def fixture(text):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    handle.write(text)
    handle.close()
    return handle.name


class JustifyTests(unittest.TestCase):
    def test_selection_tie_reference_and_configuration(self):
        path = fixture("a,b,Performance-\n1,0,10\n0,1,5\n1,1,5\n")
        try:
            result = analyze(path)
            self.assertEqual(result["recommendation"]["csv_record"], 3)
            self.assertEqual(result["recommendation"]["configuration"], {"a": "0", "b": "1"})
            self.assertEqual(result["absolute_difference"], 5)
            self.assertEqual(result["improvement_percent"], 50)
            text = render(result)
            self.assertIn("Measured result", text)
            self.assertIn("Historical reference", text)
        finally:
            os.unlink(path)

    def test_missing_and_nonfinite_values(self):
        path = fixture("a,Performance-\n1,NaN\n0,inf\n1,3\n")
        try:
            result = analyze(path)
            self.assertEqual(result["recommendation"]["performance"], 3)
            self.assertIsNone(result["reference_value"])
            self.assertIn("unavailable", render(result))
        finally:
            os.unlink(path)

    def test_zero_reference(self):
        path = fixture("a,Performance-\n1,0\n0,-1\n")
        try:
            result = analyze(path)
            self.assertIsNone(result["improvement_percent"])
            self.assertIn("reference outcome is zero", render(result))
        finally:
            os.unlink(path)

    def test_no_usable_records(self):
        path = fixture("a,Performance-\n1,?\n0,NaN\n")
        try:
            result = analyze(path)
            self.assertEqual(result["status"], "no_usable_records")
            self.assertIn("no usable records", render(result))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
