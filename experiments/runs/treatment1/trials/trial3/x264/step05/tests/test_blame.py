import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from source.blame import rank_settings, read_measurements, render


class BlameTests(unittest.TestCase):
    def test_ranks_slow_setting_first(self):
        data = [({"slow": 0, "fast": 1}, 10), ({"slow": 0, "fast": 0}, 11),
                ({"slow": 1, "fast": 1}, 20), ({"slow": 1, "fast": 0}, 21)]
        _, ranked = rank_settings(["slow", "fast"], data)
        self.assertEqual(ranked[0]["setting"], "slow")
        self.assertAlmostEqual(ranked[0]["delta"], 10)

    def test_read_and_render(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["flag", "Performance-"])
            writer.writerow(["0", "1.0"])
            writer.writerow(["1", "3.0"])
            path = f.name
        try:
            settings, data, target = read_measurements(path)
            self.assertEqual((settings, target), (["flag"], "Performance-"))
            self.assertIn("flag", render(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
