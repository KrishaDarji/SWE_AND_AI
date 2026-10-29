import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from justify import justify

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def make(rows):
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name) / "fixture.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    return directory, path


def row(mre, acc, pred, n="10"):
    return {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": n,
            "criterion": "squared_error", "Min_sample_leaves": "2",
            "Min_impurity_decrease": "0.1", "Max_depth": "5"}


class JustifyTests(unittest.TestCase):
    def test_policy_ties_identifier_configuration_and_medians(self):
        d, path = make([row(1, .2, .3, "first"), row(0.5, .1, .2, "winner"),
                        row(0.5, .4, .1, "tie-loses"), row("nan", 9, 9)])
        self.addCleanup(d.cleanup)
        result = justify(path)
        rec = result["recommendation"]
        self.assertEqual(rec["original_record_id"], 2)
        self.assertEqual(rec["configuration"]["N_estimators"], "tie-loses")
        self.assertEqual(rec["measured_outcomes"], {"MRE-": .5, "ACC+": .4, "PRED40+": .1})
        self.assertEqual(result["references"]["MRE-"]["valid_count"], 3)
        self.assertAlmostEqual(result["references"]["MRE-"]["value"], .5)
        self.assertIn("lowest observed MRE-", result["policy"])

    def test_invalid_outcomes_are_excluded_and_no_eligible_is_explicit(self):
        d, path = make([row("nan", 1, 1), row(1, "inf", 1)])
        self.addCleanup(d.cleanup)
        result = justify(path)
        self.assertIsNone(result["recommendation"])
        self.assertEqual(result["eligible_count"], 0)
        self.assertEqual(result["references"]["MRE-"]["valid_count"], 1)
        self.assertIn("no eligible", result["status"])


if __name__ == "__main__":
    unittest.main()
