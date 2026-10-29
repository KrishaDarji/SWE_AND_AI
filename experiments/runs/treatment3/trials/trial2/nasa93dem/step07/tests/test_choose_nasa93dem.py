import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / "source" / "choose_nasa93dem.py"
spec = importlib.util.spec_from_file_location("choose_nasa93dem", MODULE)
choose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choose)


class ChooseTests(unittest.TestCase):
    def test_real_dataset_validates_and_selects_feasible_maximum(self):
        rows = choose.load_validate(Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv")
        all_rows, feasible, selected = choose.evaluate(rows, 100)
        self.assertEqual(len(all_rows), 93)
        self.assertTrue(feasible)
        self.assertEqual(selected["value"], max(item["value"] for item in feasible))
        self.assertLessEqual(selected["effort"], 100)

    def test_no_feasible_configuration(self):
        rows = [{"idX": "x", "Effort-": "10", "Kloc+": "2"}]
        self.assertIsNone(choose.evaluate(rows, 9)[2])

    def test_missing_and_bad_data_rejected(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=["idX", "Effort-"])
            writer.writeheader()
            writer.writerow({"idX": "x", "Effort-": "bad"})
            path = handle.name
        with self.assertRaises(ValueError):
            choose.load_validate(path)


if __name__ == "__main__":
    unittest.main()
