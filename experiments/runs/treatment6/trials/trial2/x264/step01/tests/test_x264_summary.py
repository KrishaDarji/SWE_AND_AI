import csv, os, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import summarize

class SummaryTests(unittest.TestCase):
    def run_case(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            handle.write(text); handle.close(); return summarize(handle.name)
        finally:
            os.unlink(handle.name)
    def test_real_shape_and_stats(self):
        result = summarize(os.environ["X264_DATASET"])
        self.assertGreater(result["record_count"], 0); self.assertEqual(result["outcome"], "Performance-")
        self.assertEqual(result["direction"], "lower is better"); self.assertEqual(result["invalid_outcomes"], 0)
    def test_frequencies_statistics(self):
        result = self.run_case("a,b,Performance-\n1,x,2\n1,y,4\n0,x,6\n")
        self.assertEqual(result["frequencies"]["a"], {"1": 2, "0": 1})
        self.assertEqual(result["outcome_stats"]["count"], 3); self.assertEqual(result["outcome_stats"]["median"], 4)
    def test_missing_invalid(self):
        result = self.run_case("a,Performance-\n?,\n1,bad\n")
        self.assertEqual(result["missing_outcomes"], 1); self.assertEqual(result["invalid_outcomes"], 1)
    def test_empty_header_only_and_no_outcome(self):
        self.assertTrue(summarize(tempfile.NamedTemporaryFile(mode="w", delete=False).name)["missing_input"])
        self.assertEqual(self.run_case("a,b\n")["record_count"], 0)
        self.assertIsNone(self.run_case("a\n1\n")["outcome"])

if __name__ == "__main__": unittest.main()
