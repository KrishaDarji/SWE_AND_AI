import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from guess import predict  # noqa: E402


class GuessTests(unittest.TestCase):
    def make_csv(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_exact_configuration_returns_exact_measure(self):
        path = self.make_csv("a,b,Performance-\n0,0,10\n1,1,20\n")
        result = predict(path, {"a": 1, "b": 1}, k=1)
        self.assertEqual(result["predicted_time"], 20.0)
        self.assertEqual(result["neighbors_used"], 1)

    def test_prediction_is_json_serializable_and_uses_neighbors(self):
        path = self.make_csv("a,Performance-\n0,10\n1,20\n2,30\n")
        result = predict(path, {"a": 1.5}, k=2)
        self.assertAlmostEqual(result["predicted_time"], 25.0)
        json.dumps(result)

    def test_missing_setting_fails(self):
        path = self.make_csv("a,Performance-\n0,10\n")
        with self.assertRaisesRegex(ValueError, "missing settings"):
            predict(path, {})


if __name__ == "__main__":
    unittest.main()
