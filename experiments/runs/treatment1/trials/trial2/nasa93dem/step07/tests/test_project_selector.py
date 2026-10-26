import csv
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from project_selector import choose_projects, load_projects


class SelectorTests(unittest.TestCase):
    def test_budget_is_respected_and_value_is_maximized(self):
        projects = [
            {"id": "a", "effort": 4, "value": 7, "settings": {}},
            {"id": "b", "effort": 5, "value": 8, "settings": {}},
            {"id": "c", "effort": 6, "value": 10, "settings": {}},
        ]
        result = choose_projects(projects, 10)
        self.assertEqual([p["id"] for p in result["selected"]], ["a", "c"])
        self.assertEqual(result["delivered_kloc"], 17)
        self.assertLessEqual(result["total_effort"], 10)

    def test_loader_reads_required_fields(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["idX", "Effort-", "Kloc+"])
            writer.writeheader(); writer.writerow({"idX": "1", "Effort-": "2.5", "Kloc+": "3.0"})
            handle.flush()
            self.assertEqual(load_projects(handle.name)[0]["id"], "1")

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            choose_projects([], -1)


if __name__ == "__main__":
    unittest.main()
