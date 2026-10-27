import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.predict import predict_encoding_time


class PredictionTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_exact_configuration_returns_measure(self):
        path = self.write("a,b,time\n0,0,10\n1,1,20\n")
        result = predict_encoding_time(path, {"a": 1, "b": 1}, neighbors=1)
        self.assertEqual(result.value, 20.0)
        self.assertEqual(result.neighbors, 1)

    def test_mean_of_nearest_neighbors(self):
        path = self.write("a,time\n0,10\n1,20\n2,30\n")
        result = predict_encoding_time(path, {"a": 1}, neighbors=2)
        self.assertEqual(result.value, 15.0)

    def test_missing_setting_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "missing setting"):
            predict_encoding_time(self.write("a,time\n0,1\n"), {})


if __name__ == "__main__":
    unittest.main()
