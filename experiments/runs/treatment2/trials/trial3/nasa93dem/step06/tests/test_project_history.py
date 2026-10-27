import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.repository import CsvProjectRepository
from project_history.service import summarize_projects
from project_history.estimator import estimate_effort
from project_history.certifier import certify_less_costly
from project_history.portfolio import assess_proposals
from project_history.blame import explain_excessive_effort


class SummaryTests(unittest.TestCase):
    def test_blame_ranks_categorical_and_numeric_associations(self):
        projects = [
            Project("a", 2000, 1.0, 10.0, 1.0, 2.0, {"risk": "low"}),
            Project("b", 2001, 2.0, 20.0, 2.0, 3.0, {"risk": "low"}),
            Project("c", 2002, 3.0, 40.0, 3.0, 4.0, {"risk": "high"}),
            Project("d", 2003, 4.0, 80.0, 4.0, 5.0, {"risk": "high"}),
        ]
        result = explain_excessive_effort(projects)
        self.assertEqual(result["excessive_count"], 2)
        self.assertEqual(result["categorical"][0]["factor"], "risk")
        self.assertEqual(result["categorical"][0]["high_value"], "high")
        self.assertEqual(result["numeric"][0]["factor"], "Defects-")
        self.assertEqual(result["numeric"][0]["spearman"], 1.0)

    def test_blame_handles_missing_effort(self):
        result = explain_excessive_effort([Project("x", None, None, None, None, None, {})])
        self.assertEqual(result["excessive_count"], 0)
        self.assertEqual(result["categorical"], [])
    def test_arriving_proposals_are_assessed_sequentially(self):
        history = [Project("h", 2000, 10.0, 20.0, None, None, {})]
        result = assess_proposals(history, [{"id": "first", "kloc": 5}, {"id": "second", "kloc": 6}], portfolio_budget=15)
        self.assertEqual([item.status for item in result], ["approved", "not approved"])
        self.assertEqual(result[0].cumulative_effort, 10.0)
        self.assertEqual(result[1].cumulative_effort, 10.0)

    def test_unknown_history_is_not_approved(self):
        result = assess_proposals([Project("h", 2000, None, 20.0, None, None, {})], [{"id": "x", "kloc": 5}])
        self.assertEqual(result[0].status, "cannot assess")
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
