import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from optimizer import Project, choose, load_projects


class OptimizerTests(unittest.TestCase):
    def test_exact_choice_for_small_case(self):
        projects = [Project(1, 10, 5, 0), Project(2, 8, 4, 0), Project(3, 30, 9, 0)]
        selected, value = choose(projects, 9)
        self.assertEqual([p.project_id for p in selected], [3])
        self.assertAlmostEqual(value, 30.0)

    def test_budget_is_respected(self):
        projects = [Project(1, 10, 5, 0), Project(2, 8, 4, 0)]
        selected, _ = choose(projects, 5)
        self.assertLessEqual(sum(p.effort for p in selected), 5.0)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            choose([], -1)

    def test_dataset_loads(self):
        path = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        projects = list(load_projects(path))
        self.assertEqual(len(projects), 93)
        self.assertEqual(projects[0].project_id, 1)


if __name__ == "__main__":
    unittest.main()
