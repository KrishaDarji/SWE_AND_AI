import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import race_search


class RaceSearchTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"x": str(i), "Performance-": str(i * 2)} for i in range(16)]

    def test_baseline_budget_and_maximum(self):
        result = race_search.baseline(self.rows, 8)
        self.assertEqual(result.evaluations, 8)
        self.assertEqual(result.best_score, 14.0)

    def test_race_is_reproducible_and_uses_budget(self):
        first = race_search.race(self.rows, 8, seed=3)
        second = race_search.race(self.rows, 8, seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first.evaluations, 8)

    def test_csv_loader_strips_score(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=["x", "Performance-"])
            writer.writeheader()
            writer.writerow({"x": "1", "Performance-": " 3.5 "})
            name = handle.name
        try:
            self.assertEqual(race_search.score(race_search.load_rows(name)[0]), 3.5)
        finally:
            os.unlink(name)


if __name__ == "__main__":
    unittest.main()
