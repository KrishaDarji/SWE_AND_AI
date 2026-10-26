import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import blame


class BlameTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(blame.percentile([1, 2, 3, 4], 0.75), 3.25)

    def test_analysis_finds_expected_rows_and_factors(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "..", "..", "moot", "optimize", "process", "nasa93dem.csv")
        rows, threshold, ranked = blame.analyze(os.path.abspath(path))
        self.assertGreater(len(rows), 0)
        self.assertGreater(threshold, 0)
        self.assertEqual(len(ranked), len(rows[0]) - 2)
        self.assertIn(ranked[0][1], {"Kloc+", "Defects-", "Effort-"})

    def test_render_has_explanation(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.writer(handle)
            writer.writerow(["idX", "factor", "Effort-"])
            writer.writerow(["1", "a", "1"])
            writer.writerow(["2", "b", "10"])
            writer.writerow(["3", "b", "12"])
            writer.writerow(["4", "a", "2"])
            path = handle.name
        try:
            output = blame.render(path)
            self.assertIn("Excessive effort", output)
            self.assertIn("factor", output)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
