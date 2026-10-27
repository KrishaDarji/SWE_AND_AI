import csv
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.core import summarize_csv


class SummaryTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_default_measure_and_setting_counts(self):
        summary = summarize_csv(self.write("a,b,quality\n1,x,2\n1,y,4\n"))
        self.assertEqual(summary.rows, 2)
        self.assertEqual(summary.setting_values["a"], {"1": 2})
        self.assertEqual(summary.measures["quality"].average, 3.0)
        self.assertEqual(summary.measures["quality"].median, 3.0)

    def test_multiple_measures(self):
        summary = summarize_csv(self.write("s,q1,q2\n0,1,10\n1,3,20\n"), ["q1", "q2"])
        self.assertEqual(summary.setting_columns, ["s"])
        self.assertEqual(summary.measures["q2"].maximum, 20.0)

    def test_bad_number_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "row 2"):
            summarize_csv(self.write("s,q\n0,nope\n"))


if __name__ == "__main__":
    unittest.main()
