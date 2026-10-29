import csv
import tempfile
import unittest
from pathlib import Path

from source.nasa93_summary import load_csv, summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)

    def test_numeric_and_categorical_summaries(self):
        result = summarize(self.fixture("idX,kind,Kloc+,Effort-,Defects-,MonthsX\n1,a,2,4,6,3\n2,b,4,8,10,5\n"))
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["numeric_statistics"]["Kloc+"]["mean"], 3.0)
        self.assertEqual(result["categorical_frequencies"]["kind"], {"a": 1, "b": 1})

    def test_missing_values(self):
        result = summarize(self.fixture("a,b\n,?\n2,NA\n"))
        self.assertEqual(result["missing_values"], {"a": 1, "b": 2})

    def test_roles_and_directions(self):
        result = summarize(self.fixture("idX,team,Kloc+,Effort-,Defects-,MonthsX\n1,h,1,2,3,4\n"))
        roles = result["column_roles"]
        self.assertEqual(roles["metadata_excluded_from_future_modeling"], ["idX", "MonthsX"])
        self.assertEqual(roles["project_descriptors"], ["team"])
        self.assertEqual(roles["outcomes"], ["Kloc+", "Effort-", "Defects-"])

    def test_empty_input(self):
        path = self.fixture("a,b\n")
        result = summarize(path)
        self.assertEqual(result["record_count"], 0)
        self.assertEqual(result["inferred_types"], {"a": "unknown", "b": "unknown"})


if __name__ == "__main__":
    unittest.main()
