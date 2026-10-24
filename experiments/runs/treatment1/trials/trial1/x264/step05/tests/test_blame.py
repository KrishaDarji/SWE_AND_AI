import csv
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import blame


class BlameTests(unittest.TestCase):
    def test_effect_is_difference_of_condition_means(self):
        rows = [
            {"slow": "1", "other": "0", "Performance-": "10"},
            {"slow": "1", "other": "1", "Performance-": "14"},
            {"slow": "0", "other": "0", "Performance-": "8"},
            {"slow": "0", "other": "1", "Performance-": "9"},
        ]
        result = blame.setting_effect(rows, "slow")
        self.assertAlmostEqual(result["slowdown"], 3.5)

    def test_report_ranks_slowest_setting(self):
        rows = [
            {"slow": "1", "mild": "1", "Performance-": "20"},
            {"slow": "1", "mild": "0", "Performance-": "18"},
            {"slow": "0", "mild": "1", "Performance-": "10"},
            {"slow": "0", "mild": "0", "Performance-": "12"},
        ]
        report = blame.format_report(rows)
        self.assertIn("Most associated with slow encodes: slow", report)

    def test_cli_reads_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x", "Performance-"])
                writer.writeheader()
                writer.writerows([{"x": "1", "Performance-": "4"},
                                  {"x": "0", "Performance-": "2"}])
            self.assertEqual(blame.main([str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
