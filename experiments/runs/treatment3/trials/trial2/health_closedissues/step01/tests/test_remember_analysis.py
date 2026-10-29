import csv
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("remember_analysis", ROOT / "source" / "remember_analysis.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


DATASET = pathlib.Path("moot/optimize/hpo/Health-ClosedIssues0000.csv")


class RememberAnalysisTests(unittest.TestCase):
    def test_real_dataset_validates_and_summarizes(self):
        rows = MODULE.load_dataset(DATASET)
        self.assertEqual(len(rows), 10000)
        result = MODULE.analyze(rows)
        self.assertEqual(result["rows"], 10000)
        self.assertEqual(set(result["quality"]), set(MODULE.QUALITY_COLUMNS))
        self.assertEqual(result["configuration"]["criterion"]["unique"],
                         ["absolute_error", "poisson", "squared_error"])
        self.assertTrue(all(abs(value) <= 1 for value in result["relationships"]["numeric_pearson"].values()))

    def test_schema_error_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as handle:
            handle.write("wrong\nvalue\n")
            handle.flush()
            with self.assertRaises(ValueError):
                MODULE.load_dataset(handle.name)

    def test_missing_value_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as handle:
            writer = csv.writer(handle)
            writer.writerow(MODULE.EXPECTED_COLUMNS)
            writer.writerow([1, 2, 3, 4, "squared_error", 1, "", 2])
            handle.flush()
            with self.assertRaises(ValueError):
                MODULE.load_dataset(handle.name)


if __name__ == "__main__":
    unittest.main()
