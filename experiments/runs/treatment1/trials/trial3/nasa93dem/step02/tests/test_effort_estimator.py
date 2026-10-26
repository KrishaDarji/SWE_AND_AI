import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from source.effort_estimator import estimate_effort, load_observations, main


class EffortEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.dataset = (Path.cwd() / "moot" / "optimize" / "process" / "nasa93dem.csv").resolve()

    def test_loads_positive_observations(self):
        rows = load_observations(self.dataset)
        self.assertGreater(len(rows), 90)
        self.assertTrue(all(k > 0 and e > 0 for k, e in rows))

    def test_estimate_is_positive_and_increases_with_size(self):
        rows = load_observations(self.dataset)
        small = estimate_effort(2, rows)[0]
        large = estimate_effort(20, rows)[0]
        self.assertGreater(small, 0)
        self.assertGreater(large, small)

    def test_invalid_kloc(self):
        with self.assertRaises(ValueError):
            estimate_effort(0, [(1, 2), (2, 3)])

    def test_cli_budget_decision(self):
        result = main(["--dataset", str(self.dataset), "--kloc", "10", "--budget", "10000"])
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
