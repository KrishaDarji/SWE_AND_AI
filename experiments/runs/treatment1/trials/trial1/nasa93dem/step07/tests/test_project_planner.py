import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from project_planner import choose_project, load_projects


class ProjectPlannerTests(unittest.TestCase):
    def test_loads_dataset(self):
        path = pathlib.Path("moot/optimize/process/nasa93dem.csv")
        projects = load_projects(path)
        self.assertEqual(len(projects), 93)
        self.assertEqual(projects[0]["id"], 1)

    def test_selects_lowest_effort_feasible_project(self):
        projects = [
            {"id": 1, "effort": 10.0, "defects": 3, "months": 8, "kloc": 2},
            {"id": 2, "effort": 4.0, "defects": 9, "months": 5, "kloc": 1},
            {"id": 3, "effort": 2.0, "defects": 1, "months": 4, "kloc": 1},
        ]
        self.assertEqual(choose_project(projects, 5)["id"], 3)

    def test_tie_breaks_are_deterministic(self):
        base = {"effort": 4.0, "months": 5, "kloc": 1}
        projects = [dict(base, id=1, defects=8), dict(base, id=2, defects=2)]
        self.assertEqual(choose_project(projects, 4)["id"], 2)

    def test_rejects_impossible_budget(self):
        with self.assertRaises(ValueError):
            choose_project([{"id": 1, "effort": 2, "defects": 1, "months": 1, "kloc": 1}], 1)


if __name__ == "__main__":
    unittest.main()
