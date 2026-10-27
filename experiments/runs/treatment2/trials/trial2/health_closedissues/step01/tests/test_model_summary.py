import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_summary import summarize_rows


class SummaryTests(unittest.TestCase):
    def test_quality_and_configuration_roles(self):
        rows = [
            {"q": "1", "accuracy": "0.5", "depth": "3", "criterion": "a"},
            {"q": "3", "accuracy": "1.0", "depth": "5", "criterion": "a"},
        ]
        result = summarize_rows(rows, ["q", "accuracy"], ["depth", "criterion"])
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["quality_metrics"]["q"]["median"], 2.0)
        self.assertEqual(result["configuration"]["depth"]["maximum"], 5.0)
        self.assertEqual(result["configuration"]["criterion"]["values"], {"a": 2})

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_rows([{"q": "1"}], ["missing"], [])

    def test_empty_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_rows([], ["q"], [])


if __name__ == "__main__":
    unittest.main()
