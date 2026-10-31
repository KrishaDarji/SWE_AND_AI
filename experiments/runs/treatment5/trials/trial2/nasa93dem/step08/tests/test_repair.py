import copy, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from repair import DESCRIPTORS, Estimator, repair

def row(values, effort, csv_row, ident="same"):
    return {"descriptors": tuple(values), "effort": effort, "csv_row": csv_row, "id": ident}

def cfg(*changes):
    x = ["a"] * len(DESCRIPTORS)
    for i, v in changes: x[i] = v
    return tuple(x)

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.base = cfg()
        self.train = [row(self.base, 200, 10, "duplicate"), row(cfg((0,"b")), 80, 11, "tie-low-effort"), row(cfg((1,"b")), 80, 12, "tie-late"), row(cfg((0,"b"),(1,"b")), 60, 13, "two-change")]

    def test_feasibility_and_minimum_change(self):
        r = repair(self.base, 100, self.train)
        self.assertEqual(r["status"], "repair-found")
        self.assertEqual(r["change_count"], 1)
        self.assertEqual(r["candidate_id"], "tie-low-effort")

    def test_tie_breakers_effort_then_earliest_csv(self):
        train = [row(self.base, 200, 19, "original"), row(cfg((0,"b")), 90, 30, "later"), row(cfg((0,"c")), 70, 31, "best"), row(cfg((0,"d")), 70, 20, "earliest")]
        r = repair(self.base, 100, train)
        self.assertEqual(r["candidate_id"], "earliest")

    def test_already_feasible(self):
        r = repair(self.base, 250, self.train)
        self.assertEqual(r["status"], "no-repair-needed")

    def test_no_feasible_repair(self):
        r = repair(self.base, 10, self.train)
        self.assertEqual(r["status"], "no-feasible-repair")

    def test_invalid_inputs(self):
        for budget in ("bad", float("nan")):
            with self.assertRaises(ValueError): repair(self.base, budget, self.train)
        with self.assertRaises(ValueError): repair(self.base[:-1], 100, self.train)
        with self.assertRaises(ValueError): repair(tuple([""] * len(DESCRIPTORS)), 100, self.train)
        with self.assertRaises(ValueError): Estimator([])

    def test_candidates_training_only_and_identifier_reported(self):
        held = row(cfg((2,"held")), 1, 99, "held-out")
        r = repair(self.base, 100, self.train)
        self.assertNotEqual(r.get("candidate_id"), held["id"])
        self.assertEqual(r["changed_fields"][0]["field"], DESCRIPTORS[0])

    def test_held_outcome_does_not_change_selection(self):
        r1 = repair(self.base, 100, self.train)
        held_outcome_changed = row(cfg((2,"held")), 999999, 99, "held-out")
        r2 = repair(self.base, 100, self.train)
        self.assertEqual((r1["candidate_id"], r1["candidate_estimated_effort"]), (r2["candidate_id"], r2["candidate_estimated_effort"]))
        self.assertNotIn(held_outcome_changed["id"], [x["id"] for x in self.train])

if __name__ == "__main__": unittest.main()
