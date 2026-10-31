import math
import os
import sys
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from choose import choose

BASE = {"MRE-":"0.5", "ACC+":"0.5", "PRED40+":"0.5", "N_estimators":"10",
        "criterion":"squared_error", "Min_sample_leaves":"2",
        "Min_impurity_decrease":"0.1", "Max_depth":"4"}
def row(**changes):
    x = BASE.copy(); x.update(changes); return x

class ChooseTests(unittest.TestCase):
    def test_boundary_and_best(self):
        r = choose([row(N_estimators="100", **{"MRE-":"0.1"}), row(N_estimators="101", **{"MRE-":"0.01"})], 100)
        self.assertEqual((r["feasible"], r["total"], r["record_id"]), (1, 2, 1))
        self.assertEqual(r["outcomes"]["MRE-"], "0.1")
    def test_all_tie_breakers(self):
        rows = [row(**{"MRE-":"0.2", "ACC+":"0.8", "PRED40+":"0.1"}),
                row(**{"MRE-":"0.2", "ACC+":"0.9", "PRED40+":"0.1"}),
                row(**{"MRE-":"0.2", "ACC+":"0.9", "PRED40+":"0.2"}),
                row(**{"MRE-":"0.2", "ACC+":"0.9", "PRED40+":"0.2"})]
        self.assertEqual(choose(rows, 10)["record_id"], 3)
    def test_invalid_caps(self):
        for cap in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): choose([row()], cap)
    def test_invalid_values_and_identifier(self):
        r = choose([row(**{"MRE-":"bad"}), row(criterion=" "), row(Max_depth="nan"), row(**{"MRE-":"0.1"})], 10)
        self.assertEqual((r["feasible"], r["total"], r["record_id"]), (1, 4, 4))
    def test_empty_and_none(self):
        with self.assertRaises(LookupError): choose([], 10)
        with self.assertRaises(LookupError): choose([row(N_estimators="11")], 10)
    def test_missing_configuration(self):
        x = row(); del x["criterion"]
        with self.assertRaises(LookupError): choose([x], 10)

if __name__ == "__main__": unittest.main()
