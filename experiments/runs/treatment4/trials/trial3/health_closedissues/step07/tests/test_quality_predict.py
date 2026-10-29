import csv, sys, tempfile, unittest
from pathlib import Path
import numpy as np
SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from quality_predict import evaluate, fit_and_predict, _frame, _pipeline, MISSING

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
def make(rows):
    d = Path(tempfile.mkdtemp()) / "x.csv"
    with d.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return d
def row(i, **kw):
    x = {"MRE-": str(.1+i/100), "ACC+": str(.8-i/100), "PRED40+": str(i), "N_estimators": str(10+i), "criterion": "squared_error", "Min_sample_leaves": str(2+i), "Min_impurity_decrease": str(i/10), "Max_depth": str(5+i)}; x.update(kw); return x

class QualityTests(unittest.TestCase):
    def test_grouping_and_reproducibility(self):
        p = make([row(i) for i in range(8)] + [row(0, **{"MRE-":"99", "ACC+":"99", "PRED40+":"99"})])
        a, b = evaluate(p), evaluate(p)
        self.assertEqual(a, b); self.assertGreaterEqual(a["train_count"], 1)
    def test_outcome_exclusion_and_identical_group(self):
        rows = [row(i) for i in range(8)]; rows += [row(0, **{"MRE-":"999", "ACC+":"999", "PRED40+":"999"})]
        p = make(rows); before = evaluate(p); rows[-1].update({"MRE-":"-999", "ACC+":"-999", "PRED40+":"-999"}); q = make(rows)
        self.assertEqual(before["train_count"], evaluate(q)["train_count"])
    def test_missing_unseen_and_preprocessing(self):
        p = make([row(i, criterion="" if i == 0 else ("new" if i == 1 else "squared_error"), Max_depth="nan" if i == 2 else str(i+5)) for i in range(8)])
        result = fit_and_predict(p, {"N_estimators":"bad", "criterion":"unseen", "Min_sample_leaves":"", "Min_impurity_decrease":"inf", "Max_depth":None})
        self.assertEqual(set(result), {"MRE-", "ACC+"}); self.assertIn(MISSING, str(_frame([{"criterion":""}])[0]))
    def test_step1_regression_imports(self):
        from summarize import summarize
        self.assertEqual(summarize(make([]))["record_count"], 0)
    def test_invalid_and_insufficient_data(self):
        with self.assertRaises(ValueError): evaluate(make([row(0)]))
        with self.assertRaises(ValueError): evaluate(make([row(0, **{"MRE-":"", "ACC+":""}), row(1, **{"MRE-":"", "ACC+":""})]))

if __name__ == "__main__": unittest.main()
