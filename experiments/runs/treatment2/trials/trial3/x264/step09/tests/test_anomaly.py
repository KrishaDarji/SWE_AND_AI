import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.anomaly import detect_anomalies


class AnomalyTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_finds_high_robust_outlier(self):
        result = detect_anomalies(self.write("mode,time\na,10\na,11\na,9\na,10\na,100\n"))
        self.assertEqual([item.row_number for item in result], [6])
        self.assertIn("high", result[0].reason)

    def test_constant_data_and_invalid_threshold(self):
        self.assertEqual(detect_anomalies(self.write("mode,time\na,4\nb,4\n")), [])
        with self.assertRaisesRegex(ValueError, "threshold"):
            detect_anomalies(self.write("mode,time\na,4\n"), threshold=0)


if __name__ == "__main__":
    unittest.main()
