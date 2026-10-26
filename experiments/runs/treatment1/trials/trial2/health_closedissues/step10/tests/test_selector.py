import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from selector import historical_baseline, race_select, utilities, read_records

class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".csv"))
        fields = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
        rows = [
            ["0.9", "0.9", "0.9", "10", "a", "1", "0", "1"],
            ["0.1", "0.2", "0.2", "20", "b", "2", "1", "2"],
            ["0.5", "0.5", "0.5", "30", "c", "3", "2", "3"],
        ]
        with self.path.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(fields); w.writerows(rows)
    def tearDown(self): self.path.unlink()
    def test_utilities_invert_error_and_race(self):
        scored = utilities(read_records(self.path))
        self.assertLess(scored[1].utility, scored[0].utility)
        self.assertEqual(race_select(scored).values["N_estimators"], "10")
    def test_baseline_uses_accuracy(self):
        scored = utilities(read_records(self.path))
        self.assertEqual(historical_baseline(scored).values["N_estimators"], "10")

if __name__ == "__main__": unittest.main()
