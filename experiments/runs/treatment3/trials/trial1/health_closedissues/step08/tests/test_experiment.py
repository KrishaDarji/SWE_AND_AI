import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from experiment import load_validate, distance, choose


DATASET = Path("moot/optimize/hpo/Health-ClosedIssues0000.csv")


class ExperimentTests(unittest.TestCase):
    def test_validation_and_size(self):
        rows = load_validate(DATASET)
        self.assertEqual(len(rows), 10000)

    def test_distance(self):
        rows = load_validate(DATASET)
        self.assertEqual(distance(rows[0], rows[0]), 0)

    def test_selected_result_preserves_quality(self):
        rows = load_validate(DATASET)
        baseline, revised, changes = choose(rows)
        self.assertEqual(changes, 3)
        self.assertLess(float(revised["MRE-"]), float(baseline["MRE-"]))
        self.assertGreaterEqual(float(revised["ACC+"]), float(baseline["ACC+"]))
        self.assertGreaterEqual(float(revised["PRED40+"]), float(baseline["PRED40+"]))


if __name__ == "__main__":
    unittest.main()
