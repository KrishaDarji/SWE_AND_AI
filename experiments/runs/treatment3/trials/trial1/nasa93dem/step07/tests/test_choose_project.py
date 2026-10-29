import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("choose_project", HERE / "source" / "choose_project.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class ChooseProjectTests(unittest.TestCase):
    def test_real_dataset_returns_feasible_choice(self):
        result = module.choose(DATA)
        self.assertGreater(result["feasible_count"], 0)
        self.assertLessEqual(result["predicted"]["Effort-"], result["budget_effort"])
        self.assertEqual(set(result["settings"]), set(module.FEATURES))

    def test_loader_strips_categories_and_parses_targets(self):
        rows = module.load_rows(DATA)
        self.assertIsInstance(rows[0]["Kloc+"], float)
        self.assertEqual(rows[0]["prec"], "h")

    def test_empty_dataset_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv") as handle:
            handle.write(",".join(module.FEATURES + module.TARGETS) + "\n")
            handle.flush()
            with self.assertRaises(ValueError):
                module.load_rows(handle.name)


if __name__ == "__main__":
    unittest.main()
