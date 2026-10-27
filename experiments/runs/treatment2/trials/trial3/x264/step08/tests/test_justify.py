import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.justify import justify_configuration


class JustificationTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_recommends_lowest_mean_and_reports_margin(self):
        result = justify_configuration(self.write("mode,time\nslow,10\nslow,12\nfast,5\nfast,7\n"))
        self.assertEqual(result.configuration, {"mode": "fast"})
        self.assertEqual(result.samples, 2)
        self.assertEqual(result.margin, 5.0)
        self.assertIn("recommended", result.recommendation)

    def test_justifies_requested_configuration(self):
        result = justify_configuration(self.write("mode,score\na,1\na,2\nb,4\nb,5\n"), {"mode": "b"}, higher_is_better=True)
        self.assertEqual(result.mean, 4.5)
        self.assertEqual(result.margin, 3.0)

    def test_rejects_unobserved_configuration(self):
        with self.assertRaisesRegex(ValueError, "match"):
            justify_configuration(self.write("mode,time\na,1\n"), {"mode": "b"})


if __name__ == "__main__":
    unittest.main()
