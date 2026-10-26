import csv
import os
import tempfile
import unittest

from source.blame import analyze, percentile, point_biserial


class BlameTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_point_biserial_sign(self):
        self.assertGreater(point_biserial([False, False, True, True], [1, 2, 5, 6]), 0)

    def test_analyze_identifies_slow_setting(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["slow_setting", "neutral", "Performance-"])
            for slow, runtime in [(0, 10), (0, 11), (1, 30), (1, 31)]:
                writer.writerow([slow, 0, runtime])
            path = handle.name
        try:
            report = analyze(path)
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["results"][0]["setting"], "slow_setting")
            self.assertGreater(report["results"][0]["delta"], 0)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
