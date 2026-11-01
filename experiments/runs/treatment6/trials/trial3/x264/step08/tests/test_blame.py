import math
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class BlameTests(unittest.TestCase):
    def test_known_eta_direction_and_outcome_excluded(self):
        rows = [{"a": "0", "b": "x", "Performance-": str(y)} for y in (1, 1)] + [{"a": "1", "b": "x", "Performance-": str(y)} for y in (3, 3)]
        result = summary.blame_rows(rows)
        item = result["settings"][0]
        self.assertEqual(item["setting"], "a")
        self.assertAlmostEqual(item["association_score"], 1.0)
        self.assertEqual(item["lower_mean_groups"], [0.0])
        self.assertEqual(item["higher_mean_groups"], [1.0])
        self.assertNotIn("Performance-", [x["setting"] for x in result["settings"]])

    def test_numeric_normalization_and_per_setting_population(self):
        rows = [{"x": v, "y": "bad", "Performance-": str(p)} for v, p in (("1", 1), ("1.0", 3), ("2", 5), ("nan", 7), ("", 9))]
        item = {x["setting"]: x for x in summary.blame_rows(rows)["settings"]}["x"]
        self.assertEqual([(g["value"], g["count"]) for g in item["groups"]], [(1.0, 2), (2.0, 1)])
        self.assertEqual(item["eligible_count"], 3); self.assertEqual(item["excluded_count"], 2)
        self.assertAlmostEqual(item["reference_mean"], 3.0)

    def test_constant_empty_and_insufficient(self):
        constant = summary.blame_rows([{"x": "0", "Performance-": "2"}, {"x": "0", "Performance-": "2"}])["settings"][0]
        self.assertIsNone(constant["association_score"])
        self.assertEqual(summary.blame_rows([])["settings"], [])
        insufficient = summary.blame_rows([{"x": "0", "Performance-": "1"}, {"x": "bad", "Performance-": "nan"}])["settings"][0]
        self.assertEqual(insufficient["status"], "insufficient_data")

    def test_deterministic_ties(self):
        result = summary.blame_rows([{"z": "0", "a": "0", "Performance-": "1"}, {"z": "1", "a": "1", "Performance-": "2"}])
        self.assertEqual([x["setting"] for x in result["settings"]], ["a", "z"])

if __name__ == "__main__": unittest.main()
