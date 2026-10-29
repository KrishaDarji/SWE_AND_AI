import csv
import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "source"))
from solution import load_dataset, evaluate, one_change_candidates, PARAMETERS

DATA = os.path.join(HERE, "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv")


class SolutionTests(unittest.TestCase):
    def test_load_and_validate(self):
        rows = load_dataset(DATA)
        self.assertEqual(len(rows), 10000)
        self.assertTrue(all(isinstance(r["MRE-"], float) for r in rows))

    def test_candidates_change_exactly_one_parameter(self):
        rows = load_dataset(DATA)
        result = evaluate(DATA)
        candidates = one_change_candidates(rows, result["baseline"])
        self.assertGreaterEqual(len(candidates), 1)
        self.assertTrue(all(sum(r[p] != result["baseline"][p] for p in PARAMETERS) == 1
                            for r in candidates))

    def test_recommendation_is_measurable_improvement(self):
        result = evaluate(DATA)
        self.assertIsNotNone(result["best"])
        self.assertGreater(result["error_gain"], 0)
        self.assertEqual(result["changed_parameter"], "criterion")


if __name__ == "__main__":
    unittest.main()
