import csv
import tempfile
import unittest
from pathlib import Path

from source.optimizer import fit_model, load_rows, predict, suggest


DATASET = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class OptimizerTests(unittest.TestCase):
    def test_loads_expected_projects(self):
        self.assertEqual(len(load_rows(DATASET)), 93)

    def test_suggestion_is_one_driver_and_improves(self):
        rows = load_rows(DATASET)
        model = fit_model(rows)
        result = suggest(rows[0], model)
        self.assertIsNotNone(result["change"])
        driver, old, new = result["change"]
        self.assertIn(driver, rows[0])
        self.assertNotEqual(old, new)
        self.assertLess(result["estimated_effort"], result["baseline"])

    def test_constant_fields_are_not_recommended(self):
        rows = load_rows(DATASET)
        model = fit_model(rows)
        result = suggest(rows[0], model)
        self.assertNotIn(result["change"][0], {"ruse", "docu", "pcon", "site"})

    def test_missing_column_is_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
            handle.write("idX,Effort-\n1,2\n")
            name = handle.name
        with self.assertRaises(ValueError):
            load_rows(name)


if __name__ == "__main__":
    unittest.main()
