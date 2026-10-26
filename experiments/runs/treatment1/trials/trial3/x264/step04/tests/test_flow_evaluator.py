import io
import json
import unittest

from source.flow_evaluator import FlowEvaluator, rows_from_csv, run


class FlowEvaluatorTests(unittest.TestCase):
    def test_updates_only_on_lower_score(self):
        evaluator = FlowEvaluator("score")
        self.assertTrue(evaluator.evaluate({"x": "a", "score": "4"})["improved"])
        self.assertFalse(evaluator.evaluate({"x": "b", "score": "5"})["improved"])
        result = evaluator.evaluate({"x": "c", "score": "2"})
        self.assertEqual(result["best"]["index"], 3)
        self.assertEqual(result["best"]["configuration"]["x"], "c")

    def test_csv_is_processed_in_arrival_order(self):
        data = io.StringIO("a,b,score\n0,1,8\n1,0,3\n")
        rows = list(rows_from_csv(data))
        self.assertEqual(rows[0]["a"], "0")
        output = io.StringIO()
        run(io.StringIO("a,b,score\n0,1,8\n1,0,3\n"), output, "score")
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[-1]["best"]["score"], 3.0)
        self.assertEqual(len(messages), 3)

    def test_invalid_score_is_rejected(self):
        with self.assertRaises(ValueError):
            FlowEvaluator("score").evaluate({"score": "not-a-number"})


if __name__ == "__main__":
    unittest.main()
