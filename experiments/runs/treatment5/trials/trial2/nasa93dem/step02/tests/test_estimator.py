import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from estimator import Estimator, grouped_split, schema, evaluate

P = ["a", "b"]
def rows(values):
    return [{"a": a, "b": b, "Effort-": str(y), "Kloc+": "999", "Defects-": "1", "metaX": "x"} for a,b,y in values]

class Tests(unittest.TestCase):
    def test_exclusions_and_whitespace(self):
        rs = rows([(" x ", "q", 10), ("x", "q", 20), ("z", "q", 30)])
        self.assertEqual(schema(rs), P)
        m = Estimator(P).fit([(r, float(r["Effort-"])) for r in rs])
        self.assertGreater(m.predict_one({"a":"x", "b":"q"}), 10)

    def test_group_split_and_reproducibility(self):
        rs = rows([(str(i), "q", i) for i in range(8)])
        a,b = grouped_split(rs, P, 42)
        c,d = grouped_split(rs, P, 42)
        self.assertEqual([(x[0]["a"]) for x in a], [x[0]["a"] for x in c])
        self.assertEqual({r["a"] for r,_ in a}.isdisjoint({r["a"] for r,_ in b}), True)

    def test_missing_unseen_and_proposal_schema(self):
        m = Estimator(P).fit([(r, y) for r,y in [({"a":"?","b":" q "}, 10), ({"a":"x","b":"q"}, 20)]])
        self.assertTrue(math.isfinite(m.predict_one({"a":"?", "b":"q"})))
        self.assertEqual(m.predict_one({"a":"new", "b":"new"}), 15)
        with self.assertRaises(ValueError): m.predict_one({"a":"x"})
        with self.assertRaises(ValueError): m.predict_one({"a":"x", "b":"q", "c":"x"})

    def test_training_only_fit_and_outcome_independence(self):
        train = [({"a":"x", "b":"q"}, 10), ({"a":"z", "b":"q"}, 20)]
        m1 = Estimator(P).fit(train); m2 = Estimator(P).fit(train)
        self.assertEqual(m1.predict_one({"a":"x", "b":"q"}), m2.predict_one({"a":"x", "b":"q"}))
        self.assertEqual(m1.global_mean, 15)
        self.assertEqual(m1.predict_one({"a":"heldout", "b":"heldout"}), 15)

    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): grouped_split(rows([("x","q",1), ("x","q",2)]), P)

if __name__ == "__main__": unittest.main()
