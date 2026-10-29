import json
import math
import unittest

from source.nasa93_spot import screen


class SpotTests(unittest.TestCase):
    cols = ["idX", "Kloc+", "Effort-", "Defects-"]

    def test_unusual_values_and_counts(self):
        rows = [{"idX": 10, "Kloc+": 1, "Effort-": 2, "Defects-": 3},
                {"idX": 20, "Kloc+": 1, "Effort-": 2, "Defects-": 3},
                {"idX": 30, "Kloc+": 1, "Effort-": 2, "Defects-": 3},
                {"idX": 40, "Kloc+": 1000, "Effort-": 2, "Defects-": 3}]
        out = screen(rows, self.cols)
        self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertEqual(out["per_outcome_flag_counts"]["Kloc+"], 1)
        self.assertEqual(out["records"][-1]["record_identifier"], 40)
        self.assertEqual(out["reference_statistics_include_assessed_record"], True)

    def test_invalid_small_empty_constant_and_zero_scale(self):
        rows = [{"idX": "a", "Kloc+": -1, "Effort-": None, "Defects-": float("nan")},
                {"idX": "b", "Kloc+": 2, "Effort-": 2, "Defects-": 1}]
        out = screen(rows, self.cols)
        self.assertEqual(out["records"][0]["record_identifier"], "a")
        self.assertIsNone(out["records"][0]["outcomes"]["Kloc+"]["score"])
        self.assertEqual(screen([], self.cols)["unique_flagged_record_count"], 0)
        constant = screen([{"idX": 1, "Kloc+": 2, "Effort-": 2, "Defects-": 2}], self.cols)
        self.assertIsNone(constant["records"][0]["outcomes"]["Effort-"]["score"])

    def test_arithmetic_and_strict_json(self):
        out = screen([{"idX": 1, "Kloc+": 1, "Effort-": 2, "Defects-": 3},
                      {"idX": 2, "Kloc+": 2, "Effort-": 4, "Defects-": 6}], self.cols)
        self.assertAlmostEqual(out["records"][0]["outcomes"]["Effort-"]["score"], 1.0 / 0.5, delta=2.0)
        json.dumps(out, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
