import csv, math, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(HERE, "source"))
import blame

class BlameTests(unittest.TestCase):
    def write_rows(self, rows):
        path = os.path.join(HERE, "fixture_current.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            fields = ["MRE-"] + blame.SETTINGS
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def base(self, ys, numeric=None, criterion=None):
        numeric = numeric or [1, 2, 3, 4][:len(ys)]
        criterion = criterion or ["a"] * len(ys)
        return [{"MRE-": str(y), "N_estimators": str(numeric[i]), "criterion": criterion[i],
                 "Min_sample_leaves": "1", "Min_impurity_decrease": "1", "Max_depth": "1"}
                for i, y in enumerate(ys)]

    def test_numeric_and_categorical_known_through_analyze(self):
        y = [0., 0., 1., 1.]
        self.assertAlmostEqual(blame.eta_squared(y, [0, 0, 1, 1])[0], 1.0)
        self.assertAlmostEqual(blame.eta_squared(y, ["a", "a", "b", "b"])[0], 1.0)
        _, result = blame.analyze(self.write_rows(self.base(y, [1, 1, 2, 2], ["a", "a", "b", "b"])))
        criterion = next(x for x in result if x["setting"] == "criterion")
        self.assertAlmostEqual(criterion["score"], 1.0)
        self.assertEqual(criterion["means"], {"a": 0.0, "b": 1.0})
        self.assertEqual(criterion["direction"], "category-specific means")
    def test_excluded_and_mixed_columns(self):
        self.assertNotIn("MRE-", blame.SETTINGS)
        self.assertIn("criterion", blame.SETTINGS)
    def test_constants_missing_all_missing(self):
        self.assertEqual(blame.quantile_groups([2., 2., 2.])[0], [0, 0, 0])
        self.assertEqual(blame.eta_squared([1., 2.], [None, None])[1], 0)
        self.assertEqual(blame.number("nan"), None)
    def test_mixed_and_all_missing_numeric(self):
        groups, bounds = blame.quantile_groups([1.0, None, 3.0, float("nan")])
        self.assertEqual(groups[1], None); self.assertEqual(groups[3], None); self.assertTrue(bounds)
        rows = self.base([1, 2, 3, 4]);
        for row in rows: row["N_estimators"] = "?"
        _, result = blame.analyze(self.write_rows(rows))
        self.assertEqual(next(x for x in result if x["setting"] == "N_estimators")["n"], 0)

    def test_label_renaming_and_nonmonotonic_path(self):
        rows = self.base([0, 1, 0, 1], [1, 2, 3, 4], ["renamed_a", "renamed_a", "renamed_b", "renamed_b"])
        _, result = blame.analyze(self.write_rows(rows))
        criterion = next(x for x in result if x["setting"] == "criterion")
        self.assertEqual(criterion["direction"], "category-specific means")
        groups = [0, 1, 2, 3]
        self.assertEqual(blame.eta_squared([0, 1, 0, 1], groups, True)[4], "nonmonotonic")

    def test_no_valid_outcomes_main(self):
        path = self.write_rows(self.base(["bad", "?"]))
        from io import StringIO
        old, sys.stdout = sys.stdout, StringIO()
        try: blame.main(["blame.py", path]); output = sys.stdout.getvalue()
        finally: sys.stdout = old
        self.assertIn("No usable MRE- outcomes", output)

if __name__ == "__main__": unittest.main()
