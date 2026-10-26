import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from certify import analyze, percentile


class CertifyTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.25), 1.75)

    def test_analysis_reads_csv_and_is_deterministic(self):
        p = Path(tempfile.mkstemp(suffix=".csv")[1])
        try:
            p.write_text("Performance-\n1\n2\n3\n4\n")
            a = analyze(p, bootstrap=200)
            b = analyze(p, bootstrap=200)
            self.assertEqual(a, b)
            self.assertEqual(a["n"], 4)
            self.assertGreater(a["difference"], 0)
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()
