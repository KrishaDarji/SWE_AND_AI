import csv
import sys
from pathlib import Path

import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from experiment import analyze, load_and_validate

DATA = Path(__file__).parents[8] / "moot" / "optimize" / "hpo" / "Health-ClosedIssues0000.csv"


class ExperimentTests(unittest.TestCase):
  def test_real_dataset_validates_and_analyzes(self):
    rows, errors = load_and_validate(DATA)
    self.assertEqual(len(rows), 10000); self.assertEqual(errors, [])
    result = analyze(DATA)
    self.assertTrue(0 <= result["index"] < len(rows))
    self.assertGreater(result["anomaly_score"], 0); self.assertGreater(result["flagged_count"], 0)


  def test_missing_column_is_rejected(self):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.csv"
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["MRE-"])
            writer.writeheader(); writer.writerow({"MRE-": "0.1"})
        with self.assertRaisesRegex(ValueError, "missing columns"): load_and_validate(p)


  def test_out_of_range_metric_is_reported(self):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.csv"
        fields = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            writer.writerow(dict(zip(fields, ["1.2","0.5","0.5","10","squared_error","1","0","1"])))
        _, errors = load_and_validate(p)
        self.assertIn((0, "MRE-", "outside [0,1]"), errors)
