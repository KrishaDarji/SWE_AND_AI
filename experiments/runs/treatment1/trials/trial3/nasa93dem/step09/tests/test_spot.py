import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import spot


class SpotTests(unittest.TestCase):
    def test_outlier_ranks_first(self):
        rows = []
        for i in range(1, 7):
            rows.append({"idX": str(i), "kind": "a", "Kloc+": "10", "Effort-": "20", "Defects-": "5", "MonthsX": "10"})
        rows[-1].update({"idX": "99", "Effort-": "200", "Defects-": "50", "MonthsX": "40"})
        self.assertEqual(spot.score_rows(rows)[0]["row"]["idX"], "99")

    def test_real_file_has_rows_and_cli_output(self):
        path = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        rows = spot.load(path)
        self.assertEqual(len(rows), 93)
        out = io.StringIO()
        with redirect_stdout(out):
            spot.main([str(path), "--top", "1"])
        self.assertIn("spot_record=", out.getvalue())


if __name__ == "__main__":
    unittest.main()
