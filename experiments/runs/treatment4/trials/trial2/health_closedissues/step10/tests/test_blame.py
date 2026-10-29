import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from blame import analyze

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

class BlameTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False); self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); f.close(); return f.name
    def row(self, i, y, criterion="a"):
        return {"MRE-": str(y), "ACC+": str(1000-i), "PRED40+": str(i), "N_estimators": str(i), "criterion": criterion, "Min_sample_leaves": "1", "Min_impurity_decrease": str(i), "Max_depth": str(i)}
    def test_numeric_categorical_exclusions_and_order(self):
        rows = [self.row(i, i, "a" if i < 3 else "b") for i in range(1, 7)]
        got = analyze(self.write(rows)); self.assertEqual(got["ranking"][0], "Max_depth")
        self.assertEqual(got["excluded_outcomes"], FIELDS[:3]); self.assertEqual(got["finite_mre_count"], 6)
        self.assertEqual(got["numeric_grouping"], "none; numeric settings remain continuous")
    def test_missing_constant_and_deterministic(self):
        rows = [self.row(1, 2), self.row(2, "nan"), self.row(3, 2, "b"), self.row(4, 2, "b")]
        rows[0]["N_estimators"] = "bad"; rows[1]["N_estimators"] = ""; rows[2]["N_estimators"] = "bad"
        a = analyze(self.write(rows)); b = analyze(self.write(rows)); self.assertEqual(a["ranking"], b["ranking"])
        self.assertEqual(a["finite_mre_count"], 3); self.assertIsNone(a["marginal_descriptive_evidence"]["Min_sample_leaves"]["magnitude"])
        self.assertEqual(a["marginal_descriptive_evidence"]["N_estimators"]["evidence"]["missing_or_nonfinite"], 2)

    def test_complete_blocks_and_independent_partial_score(self):
        rows = []
        for i in range(1, 9):
            row = self.row(i, i * 2, "a" if i % 2 else "b")
            row["Min_sample_leaves"] = str(i); rows.append(row)
        got = analyze(self.write(rows))
        self.assertEqual(got["transformed_blocks"]["N_estimators"], [0, 1])
        self.assertEqual(got["transformed_blocks"]["criterion"], [8, 9])
        self.assertEqual(set(got["adjusted_partial_r2"]), set(FIELDS[3:]))
        self.assertEqual(got["primary_score"], "adjusted_partial_r2; complete transformed block removed")

    def test_missing_indicator_all_missing_and_independent_evidence(self):
        rows = [self.row(1, 1), self.row(2, 2), self.row(3, 3)]
        for row in rows: row["N_estimators"] = ""; row["criterion"] = ""
        got = analyze(self.write(rows))
        self.assertEqual(got["block_evidence"]["N_estimators"]["missing_or_nonfinite"], 3)
        self.assertIn("all values are missing", got["block_evidence"]["N_estimators"]["imputation"])
        self.assertEqual(got["block_evidence"]["N_estimators"]["block_columns"], 2)
        self.assertEqual(got["marginal_descriptive_evidence"]["criterion"]["levels"]["__MISSING__"]["count"], 3)

    def test_signed_spearman_and_criterion_summaries(self):
        rows = [self.row(i, 10 - i, "a" if i < 4 else "b") for i in range(1, 7)]
        got = analyze(self.write(rows))
        ev = got["marginal_descriptive_evidence"]["N_estimators"]
        self.assertLess(ev["signed_spearman"], 0); self.assertAlmostEqual(ev["magnitude"], abs(ev["signed_spearman"]))
        self.assertEqual(got["marginal_descriptive_evidence"]["criterion"]["levels"]["a"]["count"], 3)

if __name__ == "__main__": unittest.main()
