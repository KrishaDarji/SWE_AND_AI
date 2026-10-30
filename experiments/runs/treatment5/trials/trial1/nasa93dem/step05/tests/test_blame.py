import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from blame import _effect, _summary, rank_rows, report

class BlameTests(unittest.TestCase):
    def test_zero_variance_different_means_is_unbounded(self):
        effect = _effect(_summary([1, 1]), _summary([3, 3]))
        self.assertTrue(math.isinf(effect["score"])); self.assertLess(effect["score"], 0); self.assertIn("different", effect["reason"])
        reverse = _effect(_summary([3, 3]), _summary([1, 1]))
        self.assertTrue(math.isinf(reverse["score"])); self.assertGreater(reverse["score"], 0)
    def test_zero_variance_equal_means_is_zero(self):
        effect = _effect(_summary([2, 2]), _summary([2, 2]))
        self.assertEqual(effect["score"], 0.0); self.assertIn("equal", effect["reason"])
    def test_insufficient_observations_are_undefined(self):
        effect = _effect(_summary([1]), _summary([2, 3]))
        self.assertIsNone(effect["score"]); self.assertIn("insufficient", effect["reason"])
    def test_regular_known_standardized_score(self):
        effect = _effect(_summary([1, 3]), _summary([2, 4]))
        self.assertAlmostEqual(effect["score"], -1 / math.sqrt(2))
    def test_known_association_and_exclusions(self):
        rows = [{"signal": "a", "noise": "x", "Effort-": "1", "Kloc+": "99", "idX": "1", "fooX": "z"},
                {"signal": "a", "noise": "y", "Effort-": "2", "Kloc+": "99"},
                {"signal": "b", "noise": "x", "Effort-": "10", "Kloc+": "99"},
                {"signal": "b", "noise": "y", "Effort-": "11", "Kloc+": "99"}]
        result = rank_rows(rows); self.assertEqual([x["descriptor"] for x in result], ["signal", "noise"])
        self.assertEqual(result[0]["direction"], "higher")
        self.assertNotIn("Kloc+", [x["descriptor"] for x in result])
    def test_constants_missing_small_and_determinism(self):
        rows = [{"a": "same", "b": "?", "c": "one", "Effort-": "1"},
                {"a": "same", "b": "x", "c": "two", "Effort-": "?"},
                {"a": "same", "b": "x", "c": "one", "Effort-": "3"}]
        first = rank_rows(rows); self.assertEqual(first, rank_rows(list(reversed(rows))))
        self.assertEqual(first[0]["score"], 0.0); self.assertEqual(first[0]["descriptor"], "a")
        self.assertTrue(all(math.isfinite(x["score"]) for x in first))
    def test_whitespace_categorical_values(self):
        rows = [{"a": " high ", "Effort-": "1"}, {"a": "high", "Effort-": "2"}, {"a": " low", "Effort-": "9"}]
        self.assertEqual(rank_rows(rows)[0]["group"], "low")
    def test_unbounded_ranking_and_readable_report(self):
        rows = [{"a": "x", "b": "u", "Effort-": "1"}, {"a": "x", "b": "v", "Effort-": "1"},
                {"a": "y", "b": "u", "Effort-": "3"}, {"a": "y", "b": "v", "Effort-": "3"}]
        ranked = rank_rows(rows); self.assertEqual(ranked[0]["descriptor"], "a")
        self.assertTrue(math.isinf(ranked[0]["score"])); self.assertIn("unbounded", report(rows))
        self.assertEqual(ranked, rank_rows(list(reversed(rows))))
    def test_lower_constant_category_direction_in_rank_and_report(self):
        rows = [{"a": "low", "Effort-": "1"}, {"a": "low", "Effort-": "1"},
                {"a": "high", "Effort-": "3"}, {"a": "high", "Effort-": "3"}]
        ranked = rank_rows(rows)
        self.assertEqual(ranked[0]["group"], "low")
        self.assertEqual(ranked[0]["direction"], "lower")
        self.assertTrue(report(rows).splitlines()[-1].split(",")[2] == "lower")

if __name__ == "__main__": unittest.main()
