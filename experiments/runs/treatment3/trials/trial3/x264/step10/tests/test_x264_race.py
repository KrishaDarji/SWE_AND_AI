import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_race import DATASET, load_dataset, random_search, race_select, run


class RaceTests(unittest.TestCase):
    def test_dataset_validation(self):
        names, X, y = load_dataset(DATASET)
        self.assertEqual((len(names), len(X), len(y)), (16, 1152, 1152))
        self.assertTrue(all(set(row) <= {0, 1} for row in X))

    def test_selection_is_reproducible_and_budgeted(self):
        _, X, y = load_dataset(DATASET)
        a = race_select(X[:800], y[:800], X[800:], budget=12)
        b = race_select(X[:800], y[:800], X[800:], budget=12)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual(len(random_search(X[800:], budget=12)), 12)

    def test_end_to_end_protocol(self):
        result = run(DATASET)
        self.assertEqual(result["evaluation_budget"], 24)
        self.assertIn(result["winner_by_regret"], {"selector", "baseline", "tie"})
        for method in ("selector", "baseline"):
            self.assertGreaterEqual(result[method]["regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
