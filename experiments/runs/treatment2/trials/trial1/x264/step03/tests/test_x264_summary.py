import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import certify_improvement, predict_encoding_time, summarize_csv


class SummaryTests(unittest.TestCase):
    def test_step1_summary_and_step2_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,20\n0,1,12\n1,1,24\n", encoding="utf-8")
            self.assertEqual(summarize_csv(path)["rows"], 4)
            self.assertEqual(predict_encoding_time(path, {"a": "1", "b": "0"})["predicted_time"], 20.0)

    def test_certification_finds_credible_faster_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,20\n1,0,10\n0,0,22\n1,0,12\n0,0,19\n1,0,11\n0,0,21\n1,0,13\n0,0,20\n1,0,10\n0,0,22\n1,0,12\n", encoding="utf-8")
            result = certify_improvement(path, {"a": "0", "b": "0"}, {"a": "1", "b": "0"})
        self.assertEqual(result["pairs"], 6)
        self.assertTrue(result["credible"])
        self.assertEqual(result["p_value"], 0.015625)

    def test_certification_rejects_no_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text("a,b,Performance-\n0,0,10\n1,0,12\n", encoding="utf-8")
            result = certify_improvement(path, {"a": "0", "b": "0"}, {"a": "1", "b": "0"})
        self.assertFalse(result["credible"])
        self.assertGreater(result["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
