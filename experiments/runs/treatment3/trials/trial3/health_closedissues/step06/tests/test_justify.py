import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../source"))
from justify import load_data, main

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv"))


class JustifyTests(unittest.TestCase):
    def test_load_validation(self):
        rows = load_data(DATA)
        self.assertEqual(len(rows), 10000)
        self.assertTrue(all(0 <= r["MRE-"] <= 1 and 0 <= r["ACC+"] <= 1 for r in rows))

    def test_report_contains_evidence_and_caveat(self):
        report, result = main(DATA)
        self.assertIn("Recommendation", report)
        self.assertIn("causality", report.lower())
        self.assertEqual(set(result["metrics"]), {"MRE-", "ACC+", "PRED40+"})
        self.assertEqual(len(result["errors"]["MRE-"]), 5)


if __name__ == "__main__":
    unittest.main()
