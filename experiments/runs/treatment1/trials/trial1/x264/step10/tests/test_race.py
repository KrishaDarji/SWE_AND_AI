import csv
import tempfile
import unittest
from pathlib import Path

from race import compare, load_records, race_search, random_search


class RaceTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mktemp(suffix=".csv"))
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["x", "Performance-"])
            writer.writeheader()
            for x, score in [(0, 9), (1, 4), (2, 7), (3, 1), (4, 6), (5, 3), (6, 8), (7, 2)]:
                writer.writerow({"x": x, "Performance-": score})

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_load_and_minimize(self):
        records = load_records(self.path)
        self.assertEqual(len(records), 8)
        self.assertEqual(race_search(records, 8, 3)[0].score, 1)

    def test_deterministic_baseline(self):
        records = load_records(self.path)
        self.assertEqual(random_search(records, 4, 3)[0].score,
                         random_search(records, 4, 3)[0].score)

    def test_compare_reports_both_strategies(self):
        result = compare(self.path, budget=8)
        self.assertIn("baseline_score", result)
        self.assertIn("race_score", result)
        self.assertGreaterEqual(result["race_evaluations"], result["baseline_evaluations"])


if __name__ == "__main__":
    unittest.main()
