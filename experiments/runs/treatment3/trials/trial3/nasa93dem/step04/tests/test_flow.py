import csv
import tempfile
import unittest
from pathlib import Path

from source.flow import evaluate


DATASET = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class FlowTests(unittest.TestCase):
    def test_real_dataset_is_reproducible_and_selects_min_effort(self):
        first = evaluate(DATASET)
        second = evaluate(DATASET)
        self.assertEqual(first, second)
        self.assertEqual((first.rows_seen, first.valid_rows, first.invalid_rows), (93, 93, 0))
        self.assertEqual(first.best_proposal[0], "6")
        self.assertEqual(first.best_effort, 8.4)

    def test_invalid_incomplete_and_tie_are_deterministic(self):
        header = ["idX", "Kloc+", "Effort-", "Defects-", "MonthsX"]
        rows = [["9", "1", "2", "3", "4"], ["2", "1", "2", "3", "4"],
                ["bad", "1", "1", "3", "4"], ["3", "1", "1", "3"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            result = evaluate(path)
        self.assertEqual(result.invalid_rows, 2)
        self.assertEqual(result.incomplete_rows, 1)
        self.assertEqual(result.best_proposal[0], "2")
        self.assertEqual(result.best_effort, 2.0)


if __name__ == "__main__":
    unittest.main()
