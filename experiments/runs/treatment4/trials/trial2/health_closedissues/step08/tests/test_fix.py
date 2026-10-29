import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from fix import repair

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


class FixTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        csv.DictWriter(f, fieldnames=FIELDS).writeheader(); csv.DictWriter(f, fieldnames=FIELDS).writerows(rows)
        f.close(); self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True)); return f.name

    def row(self, mre, acc=.5, pred=.5, **changes):
        row = {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": "10", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"}
        row.update({key: str(value) for key, value in changes.items()}); return row

    def test_minimum_change_beats_larger_improvement_and_numeric_equivalence(self):
        result = repair(self.write([self.row(10), self.row(9, N_estimators="10.0", Max_depth=3), self.row(1, N_estimators=11, Max_depth=3)]))
        self.assertEqual(result["repaired"]["record_id"], 2); self.assertEqual(result["change_count"], 1)

    def test_all_tiebreakers_changed_fields_and_worsening(self):
        rows = [self.row(10, .7), self.row(9, .4, .9, criterion="b", Max_depth=3), self.row(9, .6, .1, criterion="b", Max_depth=3), self.row(9, .6, .9, criterion="b", Max_depth=4), self.row(9, .6, .9, criterion="b", Max_depth=4)]
        result = repair(self.write(rows)); self.assertEqual(result["repaired"]["record_id"], 4)
        self.assertEqual(result["changed_fields"][0]["field"], "criterion"); self.assertTrue(result["secondary_outcome_worsening"]["ACC+"])

    def test_invalid_empty_and_no_candidate(self):
        with self.assertRaises(ValueError): repair(self.write([self.row("bad")]))
        self.assertEqual(repair(self.write([]))["status"], "empty input")
        self.assertEqual(repair(self.write([self.row(1)]))["status"], "no improving candidate")

    def test_missing_or_nonfinite_baseline_secondary_outcomes_are_unknown(self):
        for field, value in (("ACC+", ""), ("ACC+", "nan"),
                             ("PRED40+", ""), ("PRED40+", "inf")):
            baseline = self.row(10); baseline[field] = value
            result = repair(self.write([baseline, self.row(9)]))
            self.assertIsNone(result["secondary_outcome_tradeoffs"][field])
            self.assertIsNone(result["secondary_outcome_worsening"][field])
            self.assertIn("prevents comparison", result["secondary_outcome_comparison_note"][field])


if __name__ == "__main__": unittest.main()
