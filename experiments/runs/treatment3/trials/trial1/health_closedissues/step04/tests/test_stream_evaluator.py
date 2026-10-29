import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from stream_evaluator import StreamEvaluator, run


def row(mre, acc, pred, n="10"):
    return {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": n,
            "criterion": "squared_error", "Min_sample_leaves": "1", "Min_impurity_decrease": "0",
            "Max_depth": "5"}


class StreamEvaluatorTests(unittest.TestCase):
    def test_updates_one_result_at_a_time(self):
        e = StreamEvaluator()
        first = e.add(row(0.8, 0.2, 0.1))
        self.assertEqual(e.best, first)
        self.assertEqual(len(e.pareto), 1)
        second = e.add(row(0.2, 0.8, 0.9))
        self.assertEqual(e.best, second)
        self.assertEqual(len(e.pareto), 1)


    def test_tradeoff_keeps_pareto_front(self):
        e = StreamEvaluator()
        e.add(row(0.1, 0.2, 0.2))
        e.add(row(0.2, 0.9, 0.9))
        self.assertEqual(len(e.pareto), 2)


    def test_real_dataset_limit(self):
        dataset = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv")
        result = run(dataset, limit=7)
        self.assertEqual(result["seen"], 7)
        self.assertGreaterEqual(result["pareto_size"], 1)


if __name__ == "__main__":
    unittest.main()
