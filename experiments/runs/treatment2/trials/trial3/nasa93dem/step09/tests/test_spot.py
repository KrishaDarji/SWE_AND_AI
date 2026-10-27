import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.model import Project
from project_history.spot import spot_inconsistent_project


class SpotTests(unittest.TestCase):
    def test_spots_large_productivity_departure(self):
        projects = [Project("a", 2000, 10, 20, None, None, {}),
                    Project("b", 2001, 10, 22, None, None, {}),
                    Project("odd", 2002, 10, 200, None, None, {})]
        result = spot_inconsistent_project(projects)
        self.assertEqual(result["project"].project_id, "odd")
        self.assertTrue(result["inconsistent"])

    def test_can_inspect_requested_project_and_handles_missing(self):
        project = Project("a", 2000, 10, 20, None, None, {})
        self.assertFalse(spot_inconsistent_project([project], "a")["inconsistent"])
        self.assertEqual(spot_inconsistent_project([project], "missing")["reason"],
                         "project not found or unusable")

    def test_requires_usable_history(self):
        result = spot_inconsistent_project([Project("x", None, None, 4, None, None, {})])
        self.assertEqual(result["reason"], "insufficient data")


if __name__ == "__main__":
    unittest.main()
