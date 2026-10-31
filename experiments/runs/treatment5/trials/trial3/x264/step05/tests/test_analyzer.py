import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyzer import rank_settings

class AnalyzerTests(unittest.TestCase):
    def test_known_association_and_direction(self):
        rows = [{"a": str(i % 2), "b": "1", "c": "?", "Performance-": str(10 + 5*(i % 2))} for i in range(8)]
        r = rank_settings(rows)
        self.assertEqual(r[0]["setting"], "a")
        self.assertAlmostEqual(r[0]["score"], 1.0)
        self.assertAlmostEqual(r[0]["direction_high_minus_low"], 5.0)
        self.assertEqual(r[1]["groups"], 1)
        self.assertEqual(r[2]["missing_count"], 8)
    def test_multivalue_missing_and_tie_order(self):
        rows = [{"z": str(i % 3), "aa": str(i % 3), "Performance-": str(i)} for i in range(6)]
        r = rank_settings(rows)
        self.assertEqual([x["setting"] for x in r], ["aa", "z"])
        self.assertEqual([g["n"] for g in r[0]["group_summaries"]], [2, 2, 2])
    def test_correlated_settings_same_score(self):
        rows = [{"x": str(i % 2), "y": str(i % 2), "Performance-": str(i % 2)} for i in range(10)]
        r = rank_settings(rows)
        self.assertAlmostEqual(r[0]["score"], r[1]["score"])
        self.assertEqual([x["setting"] for x in r], ["x", "y"])
    def test_nonfinite_is_missing(self):
        rows = [{"x": "nan", "Performance-": "1"}, {"x": "1", "Performance-": "2"}]
        self.assertEqual(rank_settings(rows)[0]["missing_count"], 1)

if __name__ == "__main__": unittest.main()
