import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow import StreamingEvaluator, run


class FlowTests(unittest.TestCase):
    def test_first_prediction_is_unknown_then_model_predicts(self):
        e = StreamingEvaluator(["a", "b"])
        self.assertIsNone(e.predict({"a": 0, "b": 1}))
        e.observe({"a": 0, "b": 1}, 10)
        self.assertIsInstance(e.predict({"a": 1, "b": 0}), float)

    def test_streaming_run_tracks_minimum(self):
        content = "a,b,Performance-\n0,1,4\n1,0,2\n1,1,3\n"
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as f:
            f.write(content); f.flush()
            out = io.StringIO()
            summary = run(f.name, out)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["best_actual"], 2.0)
        self.assertEqual(len(out.getvalue().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
