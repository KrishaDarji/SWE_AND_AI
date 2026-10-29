import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "source"))
from remember_health_closedissues import identify_columns, load_dataset, summarize


DATASET = str(Path(HERE).parents[6] / "moot/optimize/hpo/Health-ClosedIssues0000.csv")


class RememberTests(unittest.TestCase):
    def test_load_and_roles(self):
        fields, rows = load_dataset(DATASET)
        roles = identify_columns(fields, rows)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(roles["metrics"], ["MRE-", "ACC+", "PRED40+"])
        self.assertEqual(roles["configs"], ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"])

    def test_summary_has_relationships(self):
        result = summarize(DATASET)
        self.assertEqual(result["metrics"]["ACC+"]["best_direction"], "higher")
        self.assertIn("N_estimators~MRE-", result["relationships"])
        self.assertIn("criterion~ACC+", result["relationships"])


if __name__ == "__main__":
    unittest.main()
