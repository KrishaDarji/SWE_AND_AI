import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(os.path.join(HERE, "..", "source"))
sys.path.insert(0, SOURCE)
import blame


DATASET = os.path.abspath(os.path.join(HERE, "../../../../../../../../moot/optimize/process/nasa93dem.csv"))


class BlameTests(unittest.TestCase):
    def test_load_and_target(self):
        rows, headers, target, numeric, factors = blame.load_validate(DATASET)
        self.assertEqual(len(rows), 93)
        self.assertEqual(target, "Effort-")
        self.assertIn("Kloc+", factors)

    def test_analysis_is_reproducible_and_ranked(self):
        report, metrics = blame.analyze(DATASET)
        self.assertIn("association", report)
        self.assertEqual(metrics["rows"], 93)
        self.assertGreaterEqual(metrics["auc"], 0.0)
        self.assertLessEqual(metrics["auc"], 1.0)
        self.assertEqual(len(metrics["ranking"]), 27)


if __name__ == "__main__":
    unittest.main()
