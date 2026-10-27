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


if __name__ == "__main__":
    unittest.main()
