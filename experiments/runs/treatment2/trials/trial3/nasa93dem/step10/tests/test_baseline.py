import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.baseline import compare_estimators, historical_baseline
from project_history.model import Project


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.projects = [
            Project("a", 2000, 10, 20, None, None, {}),
            Project("b", 2001, 20, 60, None, None, {}),
            Project("c", 2002, 5, None, None, None, {}),
        ]

    def test_baseline_is_median_historical_effort(self):
        self.assertEqual(historical_baseline(self.projects), 40)

    def test_comparison_reports_both_predictions_and_errors(self):
        result = compare_estimators(self.projects, 10, 35)
        self.assertEqual(result["estimate"], 25)
        self.assertEqual(result["baseline"], 40)
        self.assertEqual(result["estimate_absolute_error"], 10)
        self.assertEqual(result["baseline_absolute_error"], 5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            compare_estimators(self.projects, 0, 10)
        with self.assertRaises(ValueError):
            compare_estimators(self.projects, 1, -1)


if __name__ == "__main__":
    unittest.main()
