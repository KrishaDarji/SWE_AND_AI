import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyze import association, groups_numeric

class TestAnalysis(unittest.TestCase):
    def row(self, y, **kw):
        return {"MRE-": y, **{s: kw.get(s) for s in ["N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]}}
    def test_numeric_known_and_excluded_not_used(self):
        rows = [self.row(float(x), N_estimators=x, criterion="a", ACC=x*99, **{"Max_depth": 1}) for x in range(1, 11)]
        self.assertGreater(association(rows, "N_estimators")["score"], .9)
    def test_categorical_mixed_types(self):
        rows = [self.row(y, criterion=c, N_estimators=1) for c, y in [("z", 0), ("z", 0), ("a", 10), ("a", 10)]]
        self.assertAlmostEqual(association(rows, "criterion")["score"], 1.0)
    def test_constant_missing_all_missing(self):
        rows = [self.row(y, N_estimators=1, Min_sample_leaves=None) for y in [1, 2, 3]]
        self.assertEqual(association(rows, "N_estimators")["score"], 0.0)
        self.assertEqual(association(rows, "Min_sample_leaves")["n"], 0)
    def test_groups_are_outcome_independent_and_deterministic(self):
        a = groups_numeric([5, 1, 3, 2, 4])
        self.assertEqual(a, groups_numeric([5, 1, 3, 2, 4]))
        self.assertEqual([x[0] for x in a[1]], [1, 2, 3, 4, 5])

if __name__ == "__main__": unittest.main()
