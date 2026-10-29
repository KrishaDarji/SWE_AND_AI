import csv
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from predictor import evaluate, fit_final, predict_configuration, _features

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

def make_rows(change_targets=False):
    rows = []
    for i in range(20):
        group = i // 2
        rows.append({"MRE-": str((group + 1) / 20 + (999 if change_targets else 0)), "ACC+": str(.9 - group / 30), "PRED40+": str(group / 20),
                     "N_estimators": str(10 + group), "criterion": "a" if group % 2 == 0 else "b", "Min_sample_leaves": str(group + 1),
                     "Min_impurity_decrease": str(group / 10), "Max_depth": str(group + 2)})
    return rows

class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.files = []
    def tearDown(self):
        for f in self.files: Path(f).unlink(missing_ok=True)
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False); self.files.append(f.name)
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name
    def test_excludes_pred40_grouped_reproducible_and_baselines(self):
        path = self.write(make_rows()); a = evaluate(path); b = evaluate(path)
        self.assertEqual(a["train_count"] + a["test_count"], 20); self.assertEqual(a["train_groups"] + a["test_groups"], 10)
        self.assertEqual(a["metrics"], b["metrics"]); self.assertEqual(set(a["predictor_names"]), set(FIELDS[3:])); self.assertIn("baseline_mae", a["metrics"]["MRE-"])
    def test_missing_settings_and_unseen_criterion(self):
        path = self.write(make_rows()); models = fit_final(path)
        result = predict_configuration(models, {"N_estimators": 20, "criterion": "never_seen"})
        self.assertEqual(set(result), {"MRE-", "ACC+"})
    def test_training_only_preprocessing_and_targets(self):
        original = self.write(make_rows(False)); one = evaluate(original)
        heldout = {tuple(r[f] for f in ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")) for r in one["test_rows"]}
        changed = make_rows(False)
        for row in changed:
            key = tuple(row[f] for f in ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"))
            if key in heldout: row["MRE-"] = "777"; row["ACC+"] = "888"
        two = evaluate(self.write(changed))
        self.assertEqual(one["training_mean_baselines"], two["training_mean_baselines"])
        for target in ("MRE-", "ACC+"):
            self.assertEqual(one["test_predictions"][target], two["test_predictions"][target])
    def test_missing_target_rows_are_not_used(self):
        rows = make_rows(); rows[0]["MRE-"] = ""; result = evaluate(self.write(rows)); self.assertEqual(result["train_count"] + result["test_count"], 19)

if __name__ == "__main__": unittest.main()
