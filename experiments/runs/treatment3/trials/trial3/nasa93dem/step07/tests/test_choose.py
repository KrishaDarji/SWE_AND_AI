import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from choose import choose, delivered_value, load_dataset


class ChooseTests(unittest.TestCase):
    def test_load_and_validate_real_dataset(self):
        path = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"
        rows = load_dataset(path)
        self.assertEqual(len(rows), 93)


    def test_selects_best_feasible_value(self):
        rows = [
            {"idX": "1", "Kloc+": "10", "Effort-": "100", "Defects-": "0", "MonthsX": "5"},
            {"idX": "2", "Kloc+": "30", "Effort-": "501", "Defects-": "0", "MonthsX": "5"},
            {"idX": "3", "Kloc+": "20", "Effort-": "200", "Defects-": "10", "MonthsX": "5"},
        ]
        selected, feasible = choose(rows, budget=500)
        self.assertEqual(selected["idX"], "3")
        self.assertEqual(len(feasible), 2)
        self.assertAlmostEqual(delivered_value(selected), 13.333333333333334)


    def test_no_feasible_configuration(self):
        rows = [{"idX": "1", "Kloc+": "1", "Effort-": "2", "Defects-": "0", "MonthsX": "1"}]
        self.assertIsNone(choose(rows, budget=1)[0])
