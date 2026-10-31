import math, os, sys, unittest
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "source"))
import blame

class BlameTests(unittest.TestCase):
    def test_known_association_and_order(self):
        rows = [{"a":"high" if i > 4 else "low", "b":"same", "Effort-": float(i)} for i in range(10)]
        out = blame.rank_rows(rows, ["a", "b"])
        self.assertEqual(out[0]["descriptor"], "a")
        self.assertAlmostEqual(out[0]["score"], 0.7575757576, places=6)
        self.assertEqual(out[0]["groups"], 2)
    def test_exclusion_and_categorical_whitespace(self):
        self.assertEqual(blame.clean("  h "), "h")
        self.assertIn("Effort-", blame.EXCLUDED)
        self.assertIn("Kloc+", blame.EXCLUDED)
    def test_constants_missing_small_and_ties(self):
        rows = [{"x":"z", "y":None, "Effort-": 1.0}, {"x":"z", "y":"a", "Effort-":2.0}, {"x":None, "y":"a", "Effort-":3.0}]
        out = blame.rank_rows(rows, ["x", "y"])
        self.assertEqual([x["descriptor"] for x in out], ["x", "y"])
        self.assertEqual(out[0]["score"], 0.25)
        self.assertEqual(out[0]["groups"], 1)
    def test_cliffs(self):
        self.assertAlmostEqual(blame.cliffs([3, 4], [1, 2]), 1.0)
        self.assertIsNone(blame.cliffs([], [1]))

if __name__ == "__main__": unittest.main()
