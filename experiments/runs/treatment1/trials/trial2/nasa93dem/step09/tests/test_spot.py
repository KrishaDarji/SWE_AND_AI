import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import spot


class SpotTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(spot.score_rows([]), [])

    def test_scale_adjusted_outlier_ranks_first(self):
        rows = []
        for i in range(1, 6):
            rows.append({"idX": str(i), "Kloc+": "10", "Effort-": str(100 + i),
                         "Defects-": str(200 + i), "MonthsX": str(10 + i)})
        rows.append({"idX": "6", "Kloc+": "10", "Effort-": "10000",
                     "Defects-": "200", "MonthsX": "11"})
        self.assertEqual(spot.score_rows(rows)[0][2]["idX"], "6")

    def test_cli_reads_csv(self):
        fields = ["idX", "Kloc+", "Effort-", "Defects-", "MonthsX"]
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for i in range(1, 4):
                writer.writerow({"idX": i, "Kloc+": 10, "Effort-": 100 + i,
                                 "Defects-": 200 + i, "MonthsX": 10 + i})
            path = handle.name
        try:
            self.assertEqual(spot.main([path]), 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
