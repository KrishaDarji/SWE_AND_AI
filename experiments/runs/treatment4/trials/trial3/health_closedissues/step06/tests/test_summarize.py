import csv
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from summarize import summarize  # noqa: E402


FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def fixture(tmp_path, rows):
    path = tmp_path / "fixture.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(**overrides):
    base = dict(zip(FIELDS, ["1", "2", "3", "10", "squared_error", "2", "0.1", "5"]))
    base.update(overrides)
    return base


class SummarizeTests(unittest.TestCase):
    def test_numeric_and_categorical_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(fixture(Path(directory), [row(), row(**{"MRE-": "nan", "criterion": "absolute_error"})]))
        self.assertEqual(result["numeric_statistics"]["MRE-"]["count"], 1)
        self.assertEqual(result["categorical_frequencies"]["criterion"], {"absolute_error": 1, "squared_error": 1})


    def test_missing_nonfinite_and_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(fixture(Path(directory), [row(**{"ACC+": "", "Max_depth": "inf"})]))
        self.assertEqual(result["missing_values"]["ACC+"], 1)
        self.assertEqual(result["non_finite_values"]["Max_depth"], 1)
        self.assertEqual(result["configuration_fields"][-1], "Max_depth")
        self.assertEqual(result["outcome_fields"], ["MRE-", "ACC+", "PRED40+"])


    def test_duplicate_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(fixture(Path(directory), [row(), row(), row(N_estimators="20")]))
        self.assertEqual(result["distinct_configuration_count"], 2)
        self.assertEqual(len(result["repeated_configurations"]), 1)
        self.assertEqual(result["duplicate_row_count"], 1)


    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize(fixture(Path(directory), []))
        self.assertEqual(result["record_count"], 0)
        self.assertEqual(result["numeric_statistics"]["MRE-"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
