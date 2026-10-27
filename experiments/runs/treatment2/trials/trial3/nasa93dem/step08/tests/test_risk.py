import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.risk import recommend_cheapest_risk_change


class RiskRecommendationTests(unittest.TestCase):
    def test_chooses_cheapest_lower_risk_process(self):
        projects = [
            Project("a", 2000, 10, 20, 100, None, {"process": "current"}),
            Project("b", 2001, 10, 30, 90, None, {"process": "cheap"}),
            Project("c", 2002, 10, 50, 10, None, {"process": "safe"}),
        ]
        result = recommend_cheapest_risk_change(projects, "process", "current")
        self.assertEqual(result["recommendation"]["value"], "cheap")
        self.assertLess(result["recommendation"]["risk"], result["current_risk"])

    def test_reports_no_change_when_no_lower_risk_exists(self):
        result = recommend_cheapest_risk_change([
            Project("a", 2000, 10, 20, 10, None, {"p": "a"}),
            Project("b", 2001, 10, 30, 20, None, {"p": "b"})], "p", "a")
        self.assertIsNone(result["recommendation"])

    def test_missing_current_process_is_safe(self):
        result = recommend_cheapest_risk_change([], "p", "missing")
        self.assertEqual(result["reason"], "insufficient data")


if __name__ == "__main__":
    unittest.main()
