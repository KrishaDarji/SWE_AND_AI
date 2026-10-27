import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.stream import StreamingEvaluator


class StreamingTests(unittest.TestCase):
    def test_updates_one_result_at_a_time(self):
        evaluator = StreamingEvaluator(["mode"], "time")
        self.assertTrue(evaluator.add({"mode": "slow", "time": "10"}).best)
        self.assertTrue(evaluator.add({"mode": "fast", "time": "8"}).best)
        result = evaluator.add({"mode": "slow", "time": "12"})
        self.assertEqual(result.samples, 2)
        self.assertEqual(result.mean, 11.0)
        self.assertEqual(evaluator.best_configuration(), ("fast",))

    def test_higher_is_better_and_validation(self):
        evaluator = StreamingEvaluator(["mode"], "score", higher_is_better=True)
        evaluator.add({"mode": "a", "score": 2})
        evaluator.add({"mode": "b", "score": 3})
        self.assertEqual(evaluator.best_configuration(), ("b",))
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluator.add({"mode": "c"})
        with self.assertRaisesRegex(ValueError, "numeric"):
            evaluator.add({"mode": "c", "score": "bad"})


if __name__ == "__main__":
    unittest.main()
