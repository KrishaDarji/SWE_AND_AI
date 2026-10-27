import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects
from project_history.estimator import estimate_effort
from project_history.certifier import certify_less_costly


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

    def test_effort_estimate_uses_median_productivity(self):
        projects = [
            Project("a", 2000, 10.0, 20.0, None, None, {}),
            Project("b", 2001, 20.0, 60.0, None, None, {}),
            Project("c", 2002, 5.0, 20.0, None, None, {}),
        ]
        self.assertEqual(estimate_effort(projects, 10.0), 30.0)

    def test_effort_estimate_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            estimate_effort([], 0)

    def test_certifier_uses_normalized_effort_and_requires_evidence(self):
        projects = ([Project(f"a{i}", 2000, 10.0, 10.0 + i / 10, None, None, {"process": "a"})
                     for i in range(10)] +
                    [Project(f"b{i}", 2002, 10.0, 30.0 + i / 10, None, None, {"process": "b"})
                     for i in range(10)])
        result = certify_less_costly(projects, "process", "a", "b", trials=2000)
        self.assertEqual(result["sample_a"], 10)
        self.assertEqual(result["median_a"], 1.045)
        self.assertEqual(result["verdict"], "process A is genuinely less costly")

    def test_certifier_reports_insufficient_data(self):
        result = certify_less_costly([Project("a", 2000, 1.0, 2.0, None, None, {"p": "a"})], "p", "a", "b")
        self.assertEqual(result["verdict"], "insufficient data")


if __name__ == "__main__":
    unittest.main()
