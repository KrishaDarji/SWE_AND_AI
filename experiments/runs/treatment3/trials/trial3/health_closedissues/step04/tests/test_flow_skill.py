import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow_skill import EXPECTED, RunningEvaluator, Observation, parse_row, run


class FlowTests(unittest.TestCase):
    def test_running_best_keeps_earliest_exact_tie(self):
        state = RunningEvaluator()
        first = Observation(1, (0.2, 0.5, 0.5), (1, "x", 1, 0.0, 1))
        tied = Observation(2, first.values, (2, "y", 2, 0.0, 2))
        self.assertTrue(state.observe(first))
        self.assertFalse(state.observe(tied))
        self.assertEqual(state.best.ordinal, 1)

    def test_invalid_row_is_skipped_deterministically(self):
        row = {key: "1" for key in EXPECTED}
        row["MRE-"] = "nan"
        with self.assertRaises(ValueError):
            parse_row(row, 1)

    def test_dataset_is_sequentially_consumed(self):
        dataset = Path.cwd() / "moot" / "optimize" / "hpo" / "Health-ClosedIssues0000.csv"
        evaluator, events = run(dataset)
        self.assertEqual(evaluator.seen, 10000)
        self.assertEqual(evaluator.invalid, 0)
        self.assertGreater(len(events), 0)
        self.assertIsNotNone(evaluator.best)


if __name__ == "__main__":
    unittest.main()
