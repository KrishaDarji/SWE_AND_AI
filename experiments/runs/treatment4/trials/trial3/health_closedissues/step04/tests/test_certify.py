import csv
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from certify import compare, _welch

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

def make(rows):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "x.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return directory, path

def row(criterion, mre, acc="0.5", pred="0.4"):
    return {"MRE-": str(mre), "ACC+": acc, "PRED40+": pred, "N_estimators": "10", "criterion": criterion,
            "Min_sample_leaves": "2", "Min_impurity_decrease": "0.1", "Max_depth": "5"}

class CertifyTests(unittest.TestCase):
    def test_fixed_groups_direction_and_counts(self):
        d, p = make([row("squared_error", 5), row("squared_error", 7), row("absolute_error", 2), row("absolute_error", 4), row("other", 100)])
        self.addCleanup(d.cleanup); result = compare(p)
        self.assertEqual((result["groups"]["reference_count"], result["groups"]["candidate_count"]), (2, 2))
        self.assertEqual(result["primary"]["difference"], -3.0)
    def test_welch_arithmetic_and_threshold_uncertainty(self):
        result = _welch([1, 2, 3], [0, 1, 2])
        self.assertAlmostEqual(result["degrees_of_freedom"], 4.0)
        d, p = make([row("squared_error", x) for x in [10, 11, 12]] + [row("absolute_error", x) for x in [8, 9, 10]])
        self.addCleanup(d.cleanup); result = compare(p, 1)
        self.assertTrue(result["practical_importance"]["point_estimate_beyond_threshold"])
        self.assertFalse(result["practical_importance"]["entire_interval_beyond_threshold"])
    def test_missing_insufficient_zero_variance_and_invalid_threshold(self):
        d, p = make([row("squared_error", "nan"), row("absolute_error", 1), row("absolute_error", 2)])
        self.addCleanup(d.cleanup)
        with self.assertRaises(ValueError): compare(p)
        d2, p2 = make([row("squared_error", 2), row("squared_error", 2), row("absolute_error", 1), row("absolute_error", 1)])
        self.addCleanup(d2.cleanup); self.assertTrue(compare(p2)["primary"]["zero_variance"])
        with self.assertRaises(ValueError): compare(p2, -1)
    def test_no_threshold_is_undetermined(self):
        d, p = make([row("squared_error", 2), row("squared_error", 3), row("absolute_error", 1), row("absolute_error", 2)])
        self.addCleanup(d.cleanup); self.assertIn("undetermined", compare(p)["practical_importance"])

if __name__ == "__main__": unittest.main()
