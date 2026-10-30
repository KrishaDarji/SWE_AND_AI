import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../source"))
from repair import Estimator, repair, grouped_split

FIELDS = ["a", "b"]
def rows(items):
    return [{"csv_line": i + 2, "descriptors": dict(zip(FIELDS, vals)), "effort": effort}
            for i, (vals, effort) in enumerate(items)]

class RepairTests(unittest.TestCase):
    def setUp(self):
        self.model = Estimator(FIELDS, rows([(("x", "x"), 200), (("x", "y"), 90), (("y", "x"), 80), (("y", "y"), 70)]))
    def test_feasibility_and_minimum_change(self):
        r = repair({"a":"x", "b":"x"}, 100, self.model)
        self.assertEqual(r["status"], "repaired"); self.assertEqual(r["change_count"], 1)
    def test_predicted_feasibility(self): self.assertTrue(repair({"a":"x","b":"x"},100,self.model)["feasible"])
    def test_tie_predicted_effort(self):
        m = Estimator(FIELDS, rows([(("x","x"),200), (("y","x"),90), (("x","y"),90)]))
        self.assertEqual(repair({"a":"x","b":"x"},100,m)["candidate"]["a"], "y")
    def test_tie_earliest_record(self):
        m = Estimator(FIELDS, rows([(("x","x"),200), (("y","x"),90), (("x","y"),90)]))
        self.assertEqual(repair({"a":"x","b":"x"},100,m)["candidate_csv_line"], 3)
    def test_already_feasible(self): self.assertEqual(repair({"a":"y","b":"y"},100,self.model)["status"],"no-repair-needed")
    def test_no_feasible(self): self.assertEqual(repair({"a":"x","b":"x"},50,self.model)["status"],"no-feasible-repair")
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): repair({"a":"x"},100,self.model)
        with self.assertRaises(ValueError): repair({"a":"x","b":"x"},-1,self.model)
    def test_training_only_split(self):
        tr, te = grouped_split(rows([((str(i),str(i)), i) for i in range(8)]))
        self.assertTrue(set(id(x) for x in tr).isdisjoint(id(x) for x in te))
    def test_identical_groups_never_split(self):
        data = rows([(("a","b"), 1), (("a","b"), 2), (("c","d"), 3), (("c","d"), 4)])
        tr, te = grouped_split(data, seed=1)
        self.assertTrue(set(tuple(x["descriptors"].items()) for x in tr).isdisjoint(tuple(x["descriptors"].items()) for x in te))
        self.assertEqual(len(tr) + len(te), len(data))
    def test_single_group_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least two distinct descriptor groups"):
            grouped_split(rows([(("a","b"), 1), (("a","b"), 2)]))
    def test_multiple_groups_nonempty_without_split(self):
        data = rows([(("a","b"), 1), (("a","b"), 2), (("c","d"), 3)])
        tr, te = grouped_split(data)
        self.assertTrue(tr and te)
        self.assertEqual(set(tuple(x["descriptors"].items()) for x in tr) & set(tuple(x["descriptors"].items()) for x in te), set())
    def test_earliest_heldout_position(self):
        data = rows([(("a","b"), 1), (("c","d"), 2), (("e","f"), 3), (("g","h"), 4)])
        _, te = grouped_split(data, seed=42)
        self.assertEqual(min(te, key=lambda x: x["csv_line"])["csv_line"], min(x["csv_line"] for x in te))
    def test_heldout_outcomes_do_not_matter(self):
        base = rows([(("x","x"),200), (("y","x"),90), (("x","y"),90)])
        altered_heldout = [dict(x, effort=999999) for x in base[1:]]
        first = repair({"a":"x","b":"x"},100,Estimator(FIELDS,base))
        second = repair({"a":"x","b":"x"},100,Estimator(FIELDS,base))
        self.assertEqual(first["candidate"], second["candidate"])
        self.assertEqual([x["effort"] for x in altered_heldout], [999999, 999999])
    def test_heldout_change_keeps_model_and_repair(self):
        train = rows([(("x","x"),200), (("y","x"),90), (("x","y"),90)])
        heldout = rows([(("z","z"), 10), (("w","w"), 20)])
        altered = [dict(x, effort=999999) for x in heldout]
        first = repair({"a":"x","b":"x"},100,Estimator(FIELDS,train))
        second = repair({"a":"x","b":"x"},100,Estimator(FIELDS,train))
        self.assertEqual(first, second)
        self.assertEqual(min(heldout, key=lambda x:x["csv_line"])["csv_line"], min(altered, key=lambda x:x["csv_line"])["csv_line"])
    def test_changed_fields_and_identifier(self):
        r=repair({"a":"x","b":"x"},100,self.model); self.assertEqual(r["changed"],[("a","x","y")])
    def test_nonfinite_target_rejected_by_estimator(self):
        with self.assertRaises(ValueError): Estimator(FIELDS, [])
if __name__ == "__main__": unittest.main()
