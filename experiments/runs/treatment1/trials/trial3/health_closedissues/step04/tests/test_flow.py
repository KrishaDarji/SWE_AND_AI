import io
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from flow import FlowEvaluator, dominates, evaluate, read_observations


class FlowTests(unittest.TestCase):
    def test_streaming_summary_and_best_score(self):
        data = io.StringIO("MRE-,ACC+,PRED40+,N_estimators\n0.2,0.8,0.7,10\n0.1,0.7,0.7,20\n")
        result = evaluate(data)
        self.assertEqual(result["observations"], 2)
        self.assertEqual(result["best"]["row"], 1)
        self.assertEqual(result["pareto_size"], 2)

    def test_dominance_handles_all_objectives(self):
        rows = list(read_observations(io.StringIO(
            "MRE-,ACC+,PRED40+\n0.1,0.8,0.8\n0.2,0.8,0.7\n")))
        self.assertTrue(dominates(rows[0], rows[1]))
        self.assertFalse(dominates(rows[1], rows[0]))

    def test_empty_input(self):
        self.assertEqual(evaluate(io.StringIO("MRE-,ACC+,PRED40+\n")),
                         {"observations": 0, "best": None, "pareto_size": 0})

    def test_bad_header_is_rejected(self):
        with self.assertRaises(ValueError):
            list(read_observations(io.StringIO("MRE-,ACC+\n1,2\n")))


if __name__ == "__main__":
    unittest.main()
