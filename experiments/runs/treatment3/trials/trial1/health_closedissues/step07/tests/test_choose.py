import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import choose


class ChooseTests(unittest.TestCase):
    def test_dominated_row_is_removed(self):
        rows = [{"MRE-": 2., "ACC+": .5, "PRED40+": .5},
                {"MRE-": 1., "ACC+": .6, "PRED40+": .6}]
        self.assertEqual(len(choose.nondominated(rows)), 1)

    def test_constant_normalization(self):
        self.assertEqual(choose._normalize(3., 3., 3.), 1.0)

    def test_dataset_loads_with_expected_columns(self):
        path = Path(__file__).parents[8] / "moot/optimize/hpo/Health-ClosedIssues0000.csv"
        rows = choose.load_rows(path)
        self.assertEqual(len(rows), 10000)
        self.assertTrue(all(k in rows[0] for k in choose.OBJECTIVES + choose.PARAMETERS))


if __name__ == "__main__":
    unittest.main()
