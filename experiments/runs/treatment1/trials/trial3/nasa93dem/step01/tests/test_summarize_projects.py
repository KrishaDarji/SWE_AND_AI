import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).parents[1] / "source" / "summarize_projects.py"
SPEC = importlib.util.spec_from_file_location("summarize_projects", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


DATASET = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class SummaryTests(unittest.TestCase):
    def test_real_dataset_shape_and_duration_stats(self):
        rows = MODULE.load_projects(DATASET)
        report = MODULE.summarize(rows)
        self.assertEqual(report["project_count"], 93)
        self.assertEqual(report["year_range"], [1971, 1987])
        self.assertEqual(report["outcomes"]["MonthsX"]["min"], 4.9)
        self.assertEqual(report["outcomes"]["MonthsX"]["max"], 96.4)

    def test_small_dataset_and_empty_dataset_error(self):
        header = "idX,YearX,Kloc+,Effort-,Defects-,MonthsX\n"
        body = "1,2000,2,3,4,5\n2,2001,4,6,8,10\n"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            handle.write(header + body)
            path = Path(handle.name)
        try:
            report = MODULE.summarize(MODULE.load_projects(path))
            self.assertEqual(report["project_count"], 2)
            self.assertEqual(report["outcomes"]["MonthsX"]["median"], 7.5)
        finally:
            path.unlink()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as handle:
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                MODULE.load_projects(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
