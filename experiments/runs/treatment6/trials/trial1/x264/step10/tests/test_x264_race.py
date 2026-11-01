import unittest
from source.x264_race import race, uniform_search

class RaceTests(unittest.TestCase):
    def setUp(self):
        self.c = [{"x": str(i)} for i in range(6)] + [{"x": "0"}]
    def test_equal_budget_fresh_callbacks_and_equivalence(self):
        calls = [[], []]
        def ev(i):
            return lambda c: calls[i].append(c["x"]) or float(int(c["x"]))
        r = race(self.c, ev(0), ev(1), 3, 42)
        self.assertEqual(r["tuner"]["budget_used"], 3); self.assertEqual(r["baseline"]["budget_used"], 3)
        self.assertEqual(calls[0], calls[1]); self.assertTrue(r["tuner"]["evaluation_order"] == r["baseline"]["evaluation_order"])
    def test_hidden_outcomes_duplicate_and_known_relations(self):
        r = race(self.c, lambda c: 2, lambda c: 2, 4, 42); self.assertEqual(r["relation"], "tie")
        r = race(self.c, lambda c: 1, lambda c: 2, 4, 42); self.assertEqual(r["relation"], "tuner_win")
        r = race(self.c, lambda c: 3, lambda c: 2, 4, 42); self.assertEqual(r["relation"], "tuner_loss")
        self.assertEqual(len(set(r["tuner"]["evaluation_order"])), 4)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): race([], lambda c: 1, lambda c: 1)
        with self.assertRaises(ValueError): race(self.c, lambda c: float("nan"), lambda c: 1, 1)
        with self.assertRaises(ValueError): race(self.c, lambda c: 1, lambda c: 1, 1, 42, -1)

if __name__ == "__main__": unittest.main()
