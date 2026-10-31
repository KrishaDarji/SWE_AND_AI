import csv
import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from summarize_x264 import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False, encoding="utf-8")
        handle.write(text); handle.close(); self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_summary_frequency_statistics_and_missing(self):
        path = self.fixture("a,b,Performance-\n1,x,2\n1,x,4\n?,y,?\n")
        result = summarize(path)
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["setting_frequencies"]["a"]["1"], 2)
        self.assertEqual(result["missing_values"]["a"], 1)
        self.assertEqual(result["numeric_statistics"]["Performance-"]["mean"], 3.0)
        self.assertEqual(result["numeric_statistics"]["Performance-"]["median"], 3.0)
        self.assertEqual(result["outcome_direction"], "lower is better")

    def test_empty_input(self):
        result = summarize(self.fixture("")); self.assertEqual(result["record_count"], 0)

    def test_header_only_and_all_missing(self):
        result = summarize(self.fixture("a,Performance-\n"))
        self.assertEqual(result["record_count"], 0)
        result = summarize(self.fixture("a,Performance-\n1,?\n2,NA\n"))
        self.assertEqual(result["numeric_statistics"]["Performance-"]["count"], 0)

    def test_invalid_outcome(self):
        result = summarize(self.fixture("a,Performance-\n1,bad\n1,3\n"))
        self.assertEqual(result["invalid_values"], 1)
        self.assertEqual(result["numeric_statistics"]["Performance-"]["count"], 1)

    def test_missing_outcome_column_is_explicit(self):
        with self.assertRaises(ValueError): summarize(self.fixture("a,b\n1,2\n"))


if __name__ == "__main__": unittest.main()
