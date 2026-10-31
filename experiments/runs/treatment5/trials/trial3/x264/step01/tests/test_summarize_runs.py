import csv
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from summarize_runs import summarize


class SummaryTests(unittest.TestCase):
    def write(self, text):
        file = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        file.write(text)
        file.close()
        self.addCleanup(lambda: os.unlink(file.name))
        return file.name

    def test_summary_frequency_statistics_and_outcome(self):
        result = summarize(self.write("mode,ref,Performance-\nfast,1,3\nfast,2,1\nslow,1,2\n"))
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["settings"]["mode"]["frequencies"], {"fast": 2, "slow": 1})
        self.assertEqual(result["outcome_summary"]["median"], 2.0)
        self.assertEqual(result["outcome_summary"]["direction"], "minimize")

    def test_missing_invalid_and_outcome_count(self):
        result = summarize(self.write("x,Performance-\n?,\nA,bad\nB,4\n"))
        self.assertEqual(result["missing_values"], 1)
        self.assertEqual(result["invalid_values"], 1)
        self.assertEqual(result["outcome_summary"]["count"], 1)
        self.assertEqual(result["settings"]["x"]["missing"], 1)

    def test_empty_and_header_only(self):
        self.assertEqual(summarize(self.write(""))["record_count"], 0)
        result = summarize(self.write("x,Performance-\n"))
        self.assertEqual(result["record_count"], 0)
        self.assertIsNone(result["outcome_summary"])

    def test_entirely_missing_and_invalid_outcomes(self):
        missing = summarize(self.write("x,Performance-\nA,?\nB,NA\n"))
        self.assertIsNone(missing["outcome_summary"])
        invalid = summarize(self.write("x,Performance-\nA,nope\n"))
        self.assertEqual(invalid["invalid_values"], 1)
        self.assertIn("no valid numeric", invalid["warnings"][0])


if __name__ == "__main__":
    unittest.main()
