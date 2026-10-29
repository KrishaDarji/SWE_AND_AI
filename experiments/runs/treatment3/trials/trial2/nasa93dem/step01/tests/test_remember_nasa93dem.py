import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from source.remember_nasa93dem import analyze, identify_columns, load_dataset

DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/process/nasa93dem.csv"))


class RememberNASA93DEMTests(unittest.TestCase):
    def test_dataset_load_and_column_roles(self):
        fields, rows = load_dataset(DATASET)
        features, outcomes, ids = identify_columns(fields)
        self.assertEqual(len(rows), 93)
        self.assertEqual(len(fields), 29)
        self.assertEqual(outcomes, ["Kloc+", "Effort-", "Defects-", "MonthsX"])
        self.assertEqual(ids, ["idX"])
        self.assertEqual(len(features), 24)

    def test_analysis_contains_summaries_relationships_and_caveats(self):
        result = analyze(DATASET)
        for text in ("outcome_summary Effort-", "relationship pearson", "grouped_mean", "one_hot_linear_regression", "assumptions=", "limitations=", "data_quality="):
            self.assertIn(text, result)

    def test_validation_rejects_blank_and_missing_outcome(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("idX,Kloc+\n1,\n")
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
