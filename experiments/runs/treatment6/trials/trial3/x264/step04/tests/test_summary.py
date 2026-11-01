import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))
import summary

class SummaryTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        handle.write(text); handle.close(); self.addCleanup(lambda: Path(handle.name).unlink())
        return handle.name

    def test_summary_frequencies_statistics_and_missing(self):
        result = summary.summarize(self.write("mode,ref,Performance-\na,1,10\na,2,20\nb,1,?\n"))
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["setting_frequencies"]["mode"]["a"], 2)
        self.assertEqual(result["outcome"]["statistics"]["mean"], 15.0)
        self.assertEqual(result["outcome"]["missing_count"], 1)
        self.assertEqual(result["outcome"]["direction"], "lower is better")

    def test_empty_input(self):
        result = summary.summarize(self.write(""))
        self.assertEqual(result["record_count"], 0)

    def test_header_only(self):
        result = summary.summarize(self.write("mode,Performance-\n"))
        self.assertEqual(result["outcome"]["statistics"]["count"], 0)

    def test_entirely_missing_outcomes(self):
        result = summary.summarize(self.write("mode,Performance-\na,NA\nb,\n"))
        self.assertEqual(result["outcome"]["missing_count"], 2)
        self.assertEqual(result["outcome"]["statistics"]["count"], 0)

    def test_invalid_outcomes(self):
        result = summary.summarize(self.write("mode,Performance-\na,bad\n"))
        self.assertEqual(result["outcome"]["invalid"], [{"row": 2, "value": "bad"}])

    def test_missing_outcome_header_is_explicit(self):
        with self.assertRaises(ValueError):
            summary.summarize(self.write("mode\na\n"))

if __name__ == "__main__":
    unittest.main()
