import io
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from selector import SelectionError, choose

HEADER = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"


def data(rows):
    return io.StringIO(HEADER + "\n".join(",".join(map(str, row)) for row in rows) + "\n")


class SelectorTests(unittest.TestCase):
    def test_boundaries_and_best(self):
        result = choose(data([[.3,.1,.2,100,"a",1,0,2],[.1,.2,.3,101,"b",2,0,3]]), 100)
        self.assertEqual((result["feasible_count"], result["valid_count"], result["total_count"], result["record_id"]), (1,2,2,1))
        self.assertEqual(result["outcomes"]["MRE-"], .3)

    def test_all_tiebreakers(self):
        rows = [[.1,.4,.5,1,"a",1,0,1],[.1,.5,.1,2,"b",1,0,1],[.1,.5,.2,3,"c",1,0,1],[.1,.5,.2,4,"d",1,0,1]]
        self.assertEqual(choose(data(rows), 10)["record_id"], 3)

    def test_invalid_caps(self):
        for cap in ("-1", "1.5", "inf", "nan"):
            with self.assertRaises(SelectionError): choose(data([]), cap)

    def test_missing_nonfinite_criterion_and_empty(self):
        rows = [["nan",.1,.1,1,"a",1,0,1],[.1,.1,.1,"bad","a",1,0,1],[.1,.1,.1,1,"",1,0,1]]
        with self.assertRaises(SelectionError): choose(data(rows), 1)
        with self.assertRaises(SelectionError): choose(io.StringIO(HEADER), 1)

    def test_raw_valid_feasible_counts_and_original_identifier(self):
        rows = [[.1,.1,.1,1," ",1,0,1], [.2,.2,.2,1,"?",1,0,1],
                ["bad",.2,.2,1,"a",1,0,1], [.3,.3,.3,1,"a",1,0,1]]
        result = choose(data(rows), 1)
        self.assertEqual((result["total_count"], result["valid_count"], result["feasible_count"], result["record_id"]), (4,1,1,4))
        self.assertEqual(result["outcomes"]["MRE-"], .3)

    def test_no_feasible_and_nonfinite_cap(self):
        with self.assertRaises(SelectionError): choose(data([[.1,.1,.1,101,"a",1,0,1]]), 100)
        with self.assertRaises(SelectionError): choose(data([[.1,.1,.1,1,"a",1,0,1]]), math.inf)


if __name__ == "__main__":
    unittest.main()
