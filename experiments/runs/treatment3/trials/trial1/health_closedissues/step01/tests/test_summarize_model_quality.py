import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarize_model_quality import spearman_rank_correlation, summarize


class SummarizerTests(unittest.TestCase):
    def test_rank_correlation_handles_ties(self):
        self.assertAlmostEqual(spearman_rank_correlation([1, 2, 2], [3, 1, 1]), -1.0)

    def test_summary_and_best_direction(self):
        fields = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
        rows = [
            ["2", "0.4", "0.2", "10", "a", "1", "0.1", "3"],
            ["1", "0.8", "0.1", "20", "b", "2", "0.2", "4"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            result = summarize(path)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["best_configurations"]["MRE-"]["value"], 1.0)
        self.assertEqual(result["best_configurations"]["ACC+"]["value"], 0.8)
        self.assertEqual(result["configuration_settings"]["criterion"]["unique"], 2)


if __name__ == "__main__":
    unittest.main()
