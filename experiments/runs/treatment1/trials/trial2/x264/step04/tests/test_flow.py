import io
import unittest

from flow import FlowEvaluator, evaluate


class FlowTests(unittest.TestCase):
    def test_best_updates_only_when_lower_value_arrives(self):
        evaluator = FlowEvaluator()
        first = evaluator.accept({"feature": "a", "Performance-": "10"})
        second = evaluator.accept({"feature": "b", "Performance-": "12"})
        third = evaluator.accept({"feature": "c", "Performance-": "7.5"})
        self.assertTrue(first["improved"])
        self.assertFalse(second["improved"])
        self.assertTrue(third["improved"])
        self.assertEqual(third["best_configuration"], {"feature": "c"})

    def test_streaming_csv_preserves_arrival_order(self):
        data = "feature,Performance-\na,3\nb,2\nc,4\n"
        results = list(evaluate(io.StringIO(data)))
        self.assertEqual([item["index"] for item in results], [1, 2, 3])
        self.assertEqual(results[-1]["best_index"], 2)
        self.assertEqual(results[-1]["best_performance"], 2.0)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            FlowEvaluator().accept({"feature": "a", "Performance-": "not-a-number"})


if __name__ == "__main__":
    unittest.main()
