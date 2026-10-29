import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from spot import detect, load_validate


DATA = os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/hpo/Health-ClosedIssues0000.csv")


class SpotTests(unittest.TestCase):
    def test_validation_and_shape(self):
        rows = load_validate(DATA)
        self.assertEqual(len(rows), 10000)
        self.assertEqual(rows[0]["row"], 2)

    def test_reproducible_and_complete(self):
        rows = load_validate(DATA)
        a, b = detect(rows), detect(rows)
        self.assertEqual(a, b)
        self.assertTrue(0 <= a["anomaly_score"] <= 1)
        self.assertEqual(set(a["metrics"]), {"MRE-", "ACC+", "PRED40+"})


if __name__ == "__main__":
    unittest.main()
