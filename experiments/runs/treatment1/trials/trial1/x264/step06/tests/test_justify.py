import csv
import pathlib
import tempfile
import unittest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from justify import format_justification, read_measurements, recommend


class JustifyTests(unittest.TestCase):
    def test_recommends_lowest_measurement(self):
        rows = [
            {"flag": "0", "Performance-": "10"},
            {"flag": "1", "Performance-": "8"},
        ]
        result = recommend([{"flag": r["flag"], "Performance-": float(r["Performance-"])} for r in rows])
        self.assertEqual(result["configuration"], {"flag": "1"})
        self.assertAlmostEqual(result["improvement_percent"], 11.111111, places=4)

    def test_output_contains_action_and_limitation(self):
        result = recommend([
            {"flag": "0", "Performance-": 10.0},
            {"flag": "1", "Performance-": 8.0},
        ])
        output = format_justification(result)
        self.assertIn("Administrator action", output)
        self.assertIn("not a causal guarantee", output)

    def test_csv_reader_converts_measurement(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=["flag", "Performance-"])
            writer.writeheader()
            writer.writerow({"flag": "0", "Performance-": "3.5"})
            path = handle.name
        try:
            self.assertEqual(read_measurements(path)[0]["Performance-"], 3.5)
        finally:
            pathlib.Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
