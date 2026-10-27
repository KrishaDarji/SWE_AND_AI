import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects


class SummaryTests(unittest.TestCase):
    def test_summary_aggregates_present_values(self):
        projects = [
            Project("a", 2000, 2.0, 3.0, 4.0, 5.0, {}),
            Project("b", 2001, 1.0, 7.0, None, 9.0, {}),
        ]
        result = summarize_projects(projects)
        self.assertEqual(result.project_count, 2)
        self.assertEqual(result.year_min, 2000)
        self.assertEqual(result.total_kloc, 3.0)
        self.assertEqual(result.total_effort, 10.0)
        self.assertEqual(result.total_defects, 4.0)
        self.assertEqual(result.average_months, 7.0)

    def test_empty_input_is_safe(self):
        result = summarize_projects([])
        self.assertEqual(result.project_count, 0)
        self.assertIsNone(result.average_months)

    def test_repository_reads_trimmed_values(self):
        dataset = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        project = next(CsvProjectRepository(dataset).projects())
        self.assertEqual(project.project_id, "1")
        self.assertEqual(project.year, 1979)
        self.assertEqual(project.kloc, 25.9)
        self.assertEqual(project.months, 15.3)


if __name__ == "__main__":
    unittest.main()
