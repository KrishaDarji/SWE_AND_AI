import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from model_quality_summary import classify_columns, summarize, summarize_column


class SummaryTests(unittest.TestCase):
    def test_classification_uses_metric_suffix(self):
        self.assertEqual(classify_columns(["MRE-", "ACC+", "depth"]), (["MRE-", "ACC+"], ["depth"]))

    def test_numeric_and_categorical_statistics(self):
        numeric = summarize_column("score", ["1", "3", "2"])
        categorical = summarize_column("criterion", ["a", "b", "a"])
        self.assertEqual((numeric.minimum, numeric.maximum, numeric.mean, numeric.median), (1.0, 3.0, 2.0, 2.0))
        self.assertEqual((categorical.mode, categorical.values), ("a", {"a": 2, "b": 1}))

    def test_end_to_end_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("MRE-,ACC+,depth,criterion\n0.2,0.8,3,a\n0.4,0.6,5,b\n", encoding="utf-8")
            result = summarize(path)
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(result["metric_fields"], ["MRE-", "ACC+"])
            self.assertEqual(result["configuration_fields"], ["depth", "criterion"])
            self.assertEqual(result["columns"][3]["mode"], "a")


if __name__ == "__main__":
    unittest.main()
