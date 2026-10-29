import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from choose import choose
from race import compare, eligible_records

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class RaceTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name

    def row(self, **changes):
        base = {"MRE-": "9", "ACC+": ".1", "PRED40+": ".1", "N_estimators": "10", "criterion": "x", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"}
        base.update({k: str(v) for k, v in changes.items()}); return base

    def test_reuses_selector_eligibility_and_cap(self):
        path = self.write([self.row(**{"MRE-": 2}), self.row(**{"MRE-": 1, "N_estimators": 100}), self.row(**{"MRE-": 0, "N_estimators": 101})])
        self.assertEqual(eligible_records(path), eligible_records(path, 100))
        self.assertEqual(compare(path)["selector"], choose(path)["selected"])
        self.assertEqual(compare(path)["eligible_count"], 2)

    def test_reproducibility_independent_rng_and_pairing(self):
        path = self.write([self.row(**{"MRE-": i, "ACC+": i, "PRED40+": i}) for i in range(1, 6)])
        a, b = compare(path), compare(path)
        self.assertEqual(a["trials"], b["trials"])
        self.assertEqual(len({x["baseline_record_id"] for x in a["trials"]}), len({x["baseline_record_id"] for x in b["trials"]}))
        for trial in a["trials"]:
            self.assertAlmostEqual(trial["mre_difference"], trial["baseline_outcomes"]["MRE-"] - trial["selector_outcomes"]["MRE-"])

    def test_tie_tolerance_classification(self):
        path = self.write([self.row(**{"MRE-": 1})])
        result = compare(path, tie_tolerance=1e-12)
        self.assertEqual(result["wins_ties_losses"], {"wins": 0, "ties": 10, "losses": 0})
        self.assertIn("one historical dataset", result["interpretation"])

    def test_sign_convention_known_win_loss_tie_independent_of_implementation(self):
        path = self.write([self.row(**{"MRE-": 1})])
        selected = eligible_records(path)[0]
        baselines = [dict(selected, outcomes=dict(selected["outcomes"], **{"MRE-": 2})),
                     dict(selected, outcomes=dict(selected["outcomes"], **{"MRE-": 0})),
                     dict(selected, outcomes=dict(selected["outcomes"], **{"MRE-": 1 + 5e-13}))]
        with patch("race._draw", side_effect=baselines):
            result = compare(path, seeds=(1, 2, 3), tie_tolerance=1e-12)
        differences = [2 - 1, 0 - 1, (1 + 5e-13) - 1]
        self.assertEqual([trial["mre_difference"] for trial in result["trials"]], differences)
        self.assertEqual(result["wins_ties_losses"], {"wins": 1, "ties": 1, "losses": 1})

    def test_cap_rejects_nonfinite_values_like_selector(self):
        path = self.write([self.row(**{"MRE-": 1})])
        for cap in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                eligible_records(path, cap)

if __name__ == "__main__": unittest.main()
