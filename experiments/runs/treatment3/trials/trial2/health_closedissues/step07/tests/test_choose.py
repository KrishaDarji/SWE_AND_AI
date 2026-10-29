import csv
import os
import tempfile
import unittest
from importlib.util import spec_from_file_location, module_from_spec

HERE = os.path.dirname(os.path.dirname(__file__))
spec = spec_from_file_location("choose", os.path.join(HERE, "source", "choose.py"))
choose = module_from_spec(spec); spec.loader.exec_module(choose)


class ChooseTests(unittest.TestCase):
    def test_load_dataset_and_select(self):
        p = os.path.join(os.getcwd(), "moot/optimize/hpo/Health-ClosedIssues0000.csv")
        rows, invalid, _ = choose.load_candidates(p)
        self.assertEqual(len(rows), 10000); self.assertEqual(invalid, 0)
        score, row = choose.choose(rows)
        self.assertGreaterEqual(score, 0); self.assertIn(row["criterion"], choose.CRITERIA)

    def test_invalid_and_infeasible_rows_are_skipped(self):
        p = tempfile.mktemp(suffix=".csv")
        try:
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=choose.REQUIRED); w.writeheader()
                w.writerow({k: "" for k in choose.REQUIRED})
                row = {k: "1" for k in choose.REQUIRED}; row["criterion"] = "bad"; w.writerow(row)
                row["criterion"] = "squared_error"; row["N_estimators"] = "10"; w.writerow(row)
            rows, invalid, _ = choose.load_candidates(p)
            self.assertEqual(len(rows), 1); self.assertEqual(invalid, 2)
        finally:
            os.unlink(p)

    def test_no_feasible_rows_fails(self):
        with self.assertRaises(ValueError): choose.choose([])


if __name__ == "__main__": unittest.main()
