import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_x264", ROOT / "source" / "summarize_x264.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_summary_frequencies_missing_and_outcome(self):
        path = self.write("a,b,Performance-\n1,0,10\n1,0,20\n0,1,\n")
        got = module.summarize(path)
        self.assertEqual(got["record_count"], 3)
        self.assertEqual(got["setting_frequencies"]["a"], {"0": 1, "1": 2})
        self.assertEqual(got["missing_values"]["Performance-"], 1)
        self.assertEqual(got["outcome"], {"name": "Performance-", "direction": "lower is better"})
        self.assertEqual(got["numeric_statistics"]["Performance-"]["count"], 2)

    def test_empty_input(self):
        path = self.write("")
        got = module.summarize(path)
        self.assertEqual(got["record_count"], 0)
        self.assertEqual(got["encoder_settings"], [])

    def test_all_missing_outcome(self):
        path = self.write("x,Performance-\n1,not-a-number\n")
        got = module.summarize(path)
        self.assertEqual(got["numeric_statistics"]["Performance-"], {"count": 0})


if __name__ == "__main__":
    unittest.main()
