import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "source"))
from blame_analysis import analyze, load


class BlameTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT)), "moot", "optimize", "hpo", "Health-ClosedIssues0000.csv")
        self.path = os.path.abspath(os.path.join(ROOT, "../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv"))

    def test_load_and_schema(self):
        rows = load(self.path)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(set(rows[0]), {"MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"})

    def test_analysis_outputs(self):
        rows, cutoff, high, model, ranked, associations, auc, accuracy = analyze(self.path)
        self.assertEqual(len(rows), 10000)
        self.assertGreater(cutoff, 0)
        self.assertGreaterEqual(high, 2500)
        self.assertLessEqual(high, 2510)
        self.assertEqual(len(ranked), 7)
        self.assertTrue(0 <= auc <= 1)
        self.assertTrue(0 <= accuracy <= 1)
        self.assertEqual(len(associations), 6)


if __name__ == "__main__":
    unittest.main()
