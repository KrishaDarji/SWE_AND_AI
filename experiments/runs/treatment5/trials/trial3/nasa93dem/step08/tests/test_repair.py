import copy, math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import repair

def row(values, effort, number, ident):
    data = {key: values[i] for i, key in enumerate(repair.DESCRIPTORS)}
    data.update(_target=effort, _csv_row=number, _id=ident)
    return data

def proposal(values):
    if isinstance(values, dict):
        return {key: values[key] for key in repair.DESCRIPTORS}
    return {key: values[i] for i, key in enumerate(repair.DESCRIPTORS)}

class RepairTests(unittest.TestCase):
    def setUp(self):
        a = ["a"] * 22
        b = ["b"] * 22
        c = ["c"] * 22
        self.a, self.b, self.c = a, b, c
        self.train = [row(a, 200, 10, "same"), row(b, 90, 20, "b1"), row(b, 100, 21, "b2"), row(c, 80, 30, "c1")]
        self.model = repair.fit(self.train)

    def test_feasibility_and_minimum_change(self):
        r = repair.repair(self.model, proposal(self.a), 100)
        self.assertTrue(r["feasible"]); self.assertEqual(r["status"], "repair")
        self.assertEqual(r["candidate"], proposal(self.c))

    def test_tie_breakers_predicted_then_csv(self):
        model = repair.fit([row(self.a, 200, 10, "a"), row(self.b, 90, 40, "b"), row(self.c, 80, 30, "c")])
        r = repair.repair(model, proposal(self.a), 100)
        self.assertEqual(r["candidate"], proposal(self.c))
        model = repair.fit([row(self.a, 200, 10, "a"), row(self.b, 90, 40, "b"), row(self.c, 90, 30, "c")])
        self.assertEqual(repair.repair(model, proposal(self.a), 100)["candidate"], proposal(self.c))

    def test_already_feasible_and_none(self):
        self.assertEqual(repair.repair(self.model, proposal(self.b), 100)["status"], "no-repair-needed")
        model = repair.fit([row(self.a, 200, 1, "a")])
        self.assertEqual(repair.repair(model, proposal(self.a), 100)["status"], "no-feasible-repair")

    def test_invalid_inputs(self):
        with self.assertRaises(repair.InputError): repair.repair(self.model, proposal(self.a), "x")
        with self.assertRaises(repair.InputError): repair.repair(self.model, {"prec": "a"}, 100)
        with self.assertRaises(repair.InputError): repair.fit([])

    def test_training_only_identifiers_and_heldout_target(self):
        test = row(["z"] * 22, 1, 999, "heldout")
        r1 = repair.repair(self.model, proposal(test), 100)
        altered = copy.deepcopy(self.model); altered["train"] = self.model["train"]
        self.assertNotEqual(r1.get("candidate_id"), "heldout")
        self.assertNotIn(999, [x["_csv_row"] for x in self.model["train"]])
        test["_target"] = 999999
        self.assertEqual(r1["candidate"], repair.repair(self.model, proposal(test), 100)["candidate"])
        self.assertEqual(len(r1["changed"]), 22)

if __name__ == "__main__": unittest.main()
