import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from model_summary import OnlineEvaluator, evaluate_stream


class OnlineEvaluationTests(unittest.TestCase):
    def test_updates_incumbent_one_result_at_a_time(self):
        evaluator = OnlineEvaluator(["MRE-", "ACC+"])
        first = evaluator.observe({"depth": 2}, {"MRE-": 3, "ACC+": .5})
        self.assertEqual(first["observations_seen"], 1)
        second = evaluator.observe({"depth": 3}, {"MRE-": 1, "ACC+": .8})
        self.assertEqual(second["incumbent"]["configuration"], {"depth": 3})
        third = evaluator.observe({"depth": 4}, {"MRE-": .5, "ACC+": .7})
        self.assertEqual(third["incumbent"]["configuration"], {"depth": 3})
        self.assertEqual([x["arrival_index"] for x in third["history"]], [0, 1, 2])

    def test_stream_and_cli(self):
        records = [{"configuration": {"criterion": "a"},
                    "validation_results": {"ACC+": .6}},
                   {"configuration": {"criterion": "b"},
                    "validation_results": {"ACC+": .9}}]
        self.assertEqual(evaluate_stream(records)["incumbent"]["configuration"],
                         {"criterion": "b"})
        command = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "source", "main.py"),
                   "unused.csv", "--evaluate-stream", "--quality-column", "ACC+"]
        output = subprocess.check_output(command, input="\n".join(json.dumps(x) for x in records),
                                         text=True)
        self.assertEqual(json.loads(output)["observations_seen"], 2)

    def test_validation_is_required_and_finite(self):
        evaluator = OnlineEvaluator(["ACC+"])
        with self.assertRaises(ValueError):
            evaluator.observe({}, {"ACC+": "not-a-number"})
        with self.assertRaises(ValueError):
            evaluator.observe({}, {})


if __name__ == "__main__":
    unittest.main()
