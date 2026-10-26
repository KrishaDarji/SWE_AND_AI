import unittest
from pathlib import Path

from source.race_search import baseline_search, compare, load_records, race_search


DATASET = Path(__file__).parents[8] / "moot" / "optimize" / "config" / "X264_AllMeasurements.csv"


class RaceSearchTests(unittest.TestCase):
    def test_loads_expected_shape(self):
        records = load_records(DATASET)
        self.assertEqual(len(records), 1152)
        self.assertEqual(len(records[0].config), 16)

    def test_both_strategies_use_exact_budget(self):
        records = load_records(DATASET)
        baseline, baseline_count = baseline_search(records, 32)
        raced, race_count = race_search(records, 32)
        self.assertEqual(baseline_count, 32)
        self.assertEqual(race_count, 32)
        self.assertLessEqual(baseline.score, max(r.score for r in records))
        self.assertLessEqual(raced.score, max(r.score for r in records))

    def test_seeded_results_are_reproducible(self):
        first = compare(DATASET, 64, 11)
        second = compare(DATASET, 64, 11)
        self.assertEqual(first["baseline"], second["baseline"])
        self.assertEqual(first["race"], second["race"])

    def test_rejects_bad_budget(self):
        records = load_records(DATASET)
        with self.assertRaises(ValueError):
            baseline_search(records, 0)


if __name__ == "__main__":
    unittest.main()
