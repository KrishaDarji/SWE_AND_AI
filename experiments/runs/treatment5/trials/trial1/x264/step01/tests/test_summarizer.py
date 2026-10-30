import io
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import summarizer


class SummaryTests(unittest.TestCase):
    def summary(self, text):
        return summarizer.summarize(io.StringIO(text))

    def test_summary_frequencies_statistics_and_outcome(self):
        result = self.summary("fast,ref,Performance-\n1,2,10\n1,3,20\n0,2,30\n")
        self.assertEqual(result["record_count"], 3)
        self.assertEqual(result["settings"]["fast"]["frequencies"], {"0": 1, "1": 2})
        self.assertEqual(result["outcome"]["name"], "Performance-")
        self.assertEqual(result["outcome"]["statistics"]["median"], 20.0)
        self.assertEqual(result["outcome"]["statistics"]["mean"], 20.0)

    def test_missing_invalid_and_entirely_missing_outcomes(self):
        result = self.summary("x,Performance-\na,?\nb,bad\nc,\n")
        self.assertEqual(result["outcome"]["missing_count"], 2)
        self.assertEqual(result["outcome"]["invalid_count"], 1)
        self.assertEqual(result["status"], "no_valid_outcomes")

    def test_empty_input(self):
        self.assertEqual(self.summary("")["status"], "empty_input")

    def test_header_only_input(self):
        self.assertEqual(self.summary("x,Performance-\n")["status"], "header_only")

    def test_no_outcome_column(self):
        result = self.summary("x,y\n1,2\n")
        self.assertEqual(result["status"], "no_outcome_column")

    def test_invalid_values_are_not_statistics(self):
        result = self.summary("x,Performance-\n1,nan\n2,inf\n3,4.5\n")
        self.assertEqual(result["outcome"]["statistics"]["count"], 1)
        self.assertEqual(result["outcome"]["statistics"]["min"], 4.5)


if __name__ == "__main__":
    unittest.main()
