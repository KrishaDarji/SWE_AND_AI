import os
import sys
import unittest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import race


class RaceTests(unittest.TestCase):
    def test_validate_and_protocol(self):
        rows = race.load_validate(os.path.abspath(os.path.join(
            HERE, *([".."] * 8), "moot/optimize/hpo/Health-ClosedIssues0000.csv")))
        self.assertEqual(len(rows), 10000)
        records, summary = race.evaluate(rows, folds=2)
        self.assertEqual(len(records), 4)
        self.assertEqual(set(summary), {"treatment3_selector", "historical_baseline"})

    def test_utility_direction(self):
        self.assertGreater(race.utility({"MRE-": 0, "ACC+": 1, "PRED40+": 1}),
                           race.utility({"MRE-": 1, "ACC+": 0, "PRED40+": 0}))


if __name__ == "__main__":
    unittest.main()
