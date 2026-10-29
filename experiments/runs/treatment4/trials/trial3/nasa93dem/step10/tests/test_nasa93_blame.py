import math
import unittest

from source.nasa93_blame import rank_associations


class BlameTests(unittest.TestCase):
    columns = ["idX", "strong", "constant", "missing", "Kloc+", "Effort-", "Defects-"]

    def test_known_association_exclusions_and_categorical_handling(self):
        rows = [{"idX": i, "strong": "h" if i > 2 else "l", "constant": "x",
                 "missing": None if i == 1 else "a", "Effort-": 10 if i <= 2 else 30,
                 "Kloc+": 99, "Defects-": 1} for i in range(1, 5)]
        result = rank_associations(rows, self.columns)
        self.assertEqual(result["predictors"], ["strong", "constant", "missing"])
        self.assertEqual(result["rankings"][0]["descriptor"], "strong")
        self.assertAlmostEqual(result["rankings"][0]["score"], 1.0)
        self.assertEqual(result["rankings"][-1]["descriptor"], "constant")
        self.assertEqual(result["rankings"][0]["groups"]["h"]["count"], 2)
        self.assertEqual(result["rankings"][0]["groups"]["h"]["mean"], 30.0)

    def test_constants_missing_small_groups_and_ties_are_safe_and_deterministic(self):
        rows = [{"a": "x", "b": "x", "c": None, "Effort-": 5}, {"a": "x", "b": "y", "c": None, "Effort-": 5}]
        a = rank_associations(rows, ["a", "b", "c", "Effort-"])
        b = rank_associations(rows, ["a", "b", "c", "Effort-"])
        self.assertEqual(a, b)
        self.assertEqual([x["descriptor"] for x in a["rankings"]], ["a", "b", "c"])
        self.assertTrue(all(math.isfinite(x["score"]) and x["score"] == 0 for x in a["rankings"]))

    def test_invalid_effort_is_ignored_and_empty_is_rejected(self):
        result = rank_associations([{"a": "x", "Effort-": 1}, {"a": "y", "Effort-": float("nan")}], ["a", "Effort-"])
        self.assertEqual(result["row_count"], 1)
        with self.assertRaises(ValueError):
            rank_associations([], ["a", "Effort-"])

    def test_nominal_categories_do_not_infer_alphabetical_rating_order(self):
        rows = [{"rating": "a", "Effort-": 30}, {"rating": "a", "Effort-": 30},
                {"rating": "z", "Effort-": 10}, {"rating": "z", "Effort-": 10}]
        item = rank_associations(rows, ["rating", "Effort-"])["rankings"][0]
        self.assertEqual(item["highest_mean"], {"category": "a", "effort": 30.0})
        self.assertEqual(item["lowest_mean"], {"category": "z", "effort": 10.0})
        self.assertNotIn("higher-rated", item["direction"])
        self.assertNotIn("lower-rated", item["direction"])

    def test_equal_category_means_have_no_directional_contrast(self):
        item = rank_associations([{"rating": "a", "Effort-": 10}, {"rating": "z", "Effort-": 10}],
                                 ["rating", "Effort-"])["rankings"][0]
        self.assertEqual(item["direction"], "no directional contrast (equal category means)")


if __name__ == "__main__":
    unittest.main()
