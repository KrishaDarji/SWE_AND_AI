import unittest
from source.x264_choose import choose, audit

class ChooseTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"x": str(i)} for i in range(5)]
        self.values = {str(i): float(10 - i) for i in range(5)}

    def test_budget_membership_and_hidden_callback(self):
        calls = []
        def evaluate(config):
            calls.append(config["x"])
            return self.values[config["x"]]
        result = choose(self.candidates, evaluate, 2, 42)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["budget_used"], 2)
        self.assertIn(result["recommended_identifier"], result["evaluation_order"])
        self.assertEqual(result["recommendation"]["outcome"], min(self.values[x] for x in calls))

    def test_reproducible_duplicate_free_and_unevaluated_isolation(self):
        candidates = self.candidates + [{"x": "0"}]
        seen = []
        a = choose(candidates, lambda c: seen.append(c["x"]) or self.values[c["x"]], 20, 42)
        self.assertEqual(len(seen), 5)
        self.assertEqual(a, choose(candidates, lambda c: self.values[c["x"]], 20, 42))
        changed = list(self.values.values()); changed[0] = -999
        b = choose(self.candidates, lambda c: changed[int(c["x"])], 2, 42)
        self.assertEqual(b["evaluation_order"], a["evaluation_order"][:2])

    def test_invalid_inputs_and_overlarge_budget(self):
        for budget in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): choose(self.candidates, lambda c: 1, budget)
        with self.assertRaises(ValueError): choose([], lambda c: 1, 1)
        with self.assertRaises(ValueError): choose(self.candidates, lambda c: float("nan"), 1)
        result = choose(self.candidates, lambda c: 1, 99)
        self.assertEqual(result["budget_used"], 5)

    def test_audit_is_retrospective_and_ties_are_identifier_ordered(self):
        result = choose(self.candidates[:2], lambda c: 3, 2)
        frozen = result["recommendation"]
        self.assertEqual(frozen["record_id"], min(result["evaluation_order"]))
        self.assertEqual(audit(self.candidates[:2], [9, 1], frozen)["observed_minimum"]["record_id"], 1)
        self.assertEqual(frozen, result["recommendation"])

if __name__ == "__main__": unittest.main()
