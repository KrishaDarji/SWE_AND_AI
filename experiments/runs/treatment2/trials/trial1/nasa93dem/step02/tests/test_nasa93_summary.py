import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nasa93_summary", ROOT / "source" / "nasa93_summary.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class SummaryTests(unittest.TestCase):
    def test_loads_all_historical_projects(self):
        projects = module.load_projects(DATA)
        self.assertEqual(93, len(projects))
        self.assertEqual("1", projects[0]["id"])

    def test_outcomes_and_derived_rates_are_present(self):
        result = module.summarize_projects(DATA)
        self.assertEqual(93, result["project_count"])
        self.assertEqual(["Kloc+", "Effort-", "Defects-", "MonthsX"],
                         [item["field"] for item in result["outcomes"]])
        self.assertTrue(all("effort_per_kloc" in row for row in result["projects"]))
        self.assertGreater(result["outcomes"][0]["mean"], 0)

    def test_result_is_json_serializable(self):
        json.dumps(module.summarize_projects(DATA))

    def test_estimate_has_range_and_requires_budget_for_approval(self):
        result = module.estimate_effort(DATA, 10)
        self.assertEqual("needs_budget_review", result["approval"])
        self.assertLess(result["planning_low"], result["effort"])
        self.assertLess(result["effort"], result["planning_high"])

    def test_budget_approval_uses_upper_planning_estimate(self):
        estimate = module.estimate_effort(DATA, 10)
        approved = module.estimate_effort(DATA, 10, estimate["planning_high"])
        rejected = module.estimate_effort(DATA, 10, estimate["planning_high"] - 0.01)
        self.assertEqual("approve", approved["approval"])
        self.assertEqual("reject", rejected["approval"])

    def test_estimate_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            module.estimate_effort(DATA, 0)


if __name__ == "__main__":
    unittest.main()
