import csv
import json
import os
import tempfile
import unittest

from source.x264_summary import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile("w", delete=False, newline="", suffix=".csv")
        handle.write(text); handle.close(); self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_summary_frequencies_statistics_and_missing(self):
        result = summarize(self.fixture("a,b,Performance-\n1,x,2\n1,y,4\n?,x,?\n"))
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["settings"]["a"]["1"], 2)
        self.assertEqual(result["outcomes"]["Performance-"]["count"], 2)
        self.assertEqual(result["outcomes"]["Performance-"]["mean"], 3)
        self.assertEqual(result["missing_values"]["Performance-"], 1)

    def test_outcome_identification_and_invalid_values(self):
        result = summarize(self.fixture("setting,Performance-\nx,bad\ny,NaN\nz,3\n"))
        self.assertEqual(result["outcome_identification"]["names"], ["Performance-"])
        self.assertEqual(result["invalid_outcomes"]["Performance-"], 2)
        self.assertEqual(result["outcomes"]["Performance-"]["min"], 3)

    def test_empty_and_header_only(self):
        self.assertIn("missing header", summarize(self.fixture(""))["errors"])
        result = summarize(self.fixture("a,Performance-\n"))
        self.assertEqual(result["record_count"], 0)
        self.assertIsNone(result["outcomes"]["Performance-"]["mean"])

    def test_entirely_missing_outcome(self):
        result = summarize(self.fixture("Performance-\n?\n\n"))
        self.assertEqual(result["outcomes"]["Performance-"]["count"], 0)
        self.assertEqual(result["missing_values"]["Performance-"], 1)


if __name__ == "__main__":
    unittest.main()
