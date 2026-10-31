#!/usr/bin/env python3
import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import spot_screen


def rows(values):
    return [{"MRE-": str(a), "ACC+": str(b), "PRED40+": str(c)}
            for a, b, c in values]


class ScreenTests(unittest.TestCase):
    def test_unusual_ordinary_and_identifiers(self):
        result = spot_screen.screen_rows(rows([(1, 10, 5), (2, 10, 5),
                                               (3, 10, 5), (4, 10, 5),
                                               (100, 10, 5)]))
        flags = [x for x in result["records"] if x["flagged"]]
        self.assertEqual({x["record_id"] for x in flags}, {"data_row_5"})
        self.assertEqual(result["unique_flagged_record_count"], 1)
        self.assertEqual(result["per_outcome_flag_counts"]["MRE-"], 1)
        self.assertEqual(result["per_outcome_flag_counts"]["ACC+"], 0)

    def test_missing_nonfinite_and_json(self):
        result = spot_screen.screen_rows(rows([("?", "nan", "inf"), (1, 2, 3)]))
        undefined = [x for x in result["records"] if x["score"] is None]
        self.assertEqual(len(undefined), 3)
        self.assertNotIn("NaN", json.dumps(result, allow_nan=False))
        self.assertTrue(all("missing or nonfinite" in x["reason"] for x in undefined))

    def test_empty_and_small(self):
        result = spot_screen.screen_rows([])
        self.assertEqual(result["unique_flagged_record_count"], 0)
        one = spot_screen.screen_rows(rows([(2, 2, 2)]))
        self.assertEqual([x["score"] for x in one["records"]], [0.0, 0.0, 0.0])

    def test_constant_and_zero_scale(self):
        result = spot_screen.screen_rows(rows([(2, 3, 4), (2, 3, 4), (9, 3, 4)]))
        mre = [x for x in result["records"] if x["outcome"] == "MRE-"]
        self.assertIsNone(mre[-1]["score"])
        self.assertIn("zero robust scale", mre[-1]["reason"])
        self.assertEqual(result["per_outcome_flag_counts"]["MRE-"], 0)

    def test_scores_and_directions(self):
        result = spot_screen.screen_rows(rows([(0, 0, 0), (1, 1, 1),
                                               (2, 2, 2), (3, 3, 3), (4, 4, 4)]))
        scores = [x["score"] for x in result["records"] if x["outcome"] == "ACC+"]
        self.assertAlmostEqual(scores[0], -1.3489795003921634)
        self.assertAlmostEqual(scores[-1], 1.3489795003921634)
        self.assertTrue(all(x["threshold"] == 3.5 for x in result["records"]))


if __name__ == "__main__":
    unittest.main()
