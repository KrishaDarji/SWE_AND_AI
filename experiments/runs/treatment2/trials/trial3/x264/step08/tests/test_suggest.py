import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.suggest import suggest_setting_change


class SuggestionTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_picks_fastest_one_setting_change(self):
        path = self.write("a,b,time\n0,0,20\n0,0,22\n1,0,10\n1,0,12\n0,1,15\n0,1,16\n1,1,5\n")
        result = suggest_setting_change(path, {"a": "0", "b": "0"})
        self.assertEqual(result.setting, "a")
        self.assertEqual(result.candidate, {"a": "1", "b": "0"})
        self.assertEqual(result.improvement, 10.0)

    def test_requires_observed_baseline_and_one_change(self):
        path = self.write("a,b,time\n0,0,1\n1,1,2\n")
        with self.assertRaisesRegex(ValueError, "baseline"):
            suggest_setting_change(path, {"a": "1", "b": "0"})


if __name__ == "__main__":
    unittest.main()
