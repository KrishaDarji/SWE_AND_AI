import csv, math, pathlib, tempfile, unittest, sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa93_summary as ns

class Tests(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        f.write(text); f.close(); return f.name
    def test_real_dataset_roles_and_counts(self):
        p = pathlib.Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        h, rows, bad, _ = ns.load(str(p)); self.assertEqual(len(rows), 93); self.assertEqual(bad, [])
        self.assertEqual(ns.CATEGORICAL[0], "prec"); self.assertEqual(set(ns.OUTCOMES), {"Kloc+", "Effort-", "Defects-"})
        self.assertGreater(len(ns.summary(str(p))["categorical"]["prec"]), 1)
    def test_stats_median_variance_missing_nonfinite_invalid(self):
        p = self.write("idX,prec,Kloc+,Effort-,Defects-\n1,h,1,2,?\n2,l,3,4,NaN\n3,h,5,bad,6\n4,h,7,8,10\n")
        d = ns.summary(p); self.assertEqual(d["outcomes"]["Kloc+"]["median"], 4); self.assertAlmostEqual(d["outcomes"]["Kloc+"]["sample_variance"], 20/3)
        self.assertEqual(d["outcomes"]["Defects-"]["nonfinite"], 1); self.assertEqual(d["outcomes"]["Effort-"]["invalid"], 1)
    def test_empty_header_malformed_duplicates_positions(self):
        self.assertEqual(ns.load(self.write(""))[1], [])
        self.assertEqual(ns.load(self.write("idX,Kloc+\n"))[1], [])
        h, rows, bad, issues = ns.load(self.write("idX,Kloc+\n7,1\n7,2,extra\n7,3\n"))
        self.assertEqual([r.position for r in rows], [2, 4]); self.assertEqual(bad, [3]); self.assertTrue(issues)
        self.assertEqual(ns.summary(self.write("idX,Kloc+\n7,1\n7,2\n"))["duplicates"], {"7": 2})
    def test_exact_fields_grouping_and_reproducible_split(self):
        p = pathlib.Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        _, rows, _, _ = ns.load(str(p)); a, b = ns.split_groups(rows); c, d = ns.split_groups(rows)
        self.assertEqual([x.position for x in a], [x.position for x in c]); self.assertEqual([x.position for x in b], [x.position for x in d])
        self.assertTrue(set(ns.descriptor_key(x) for x in a).isdisjoint(set(ns.descriptor_key(x) for x in b)))
        self.assertEqual(len(ns.CATEGORICAL), 22); self.assertFalse(set(ns.CATEGORICAL) & set(ns.METADATA) & set(ns.OUTCOMES))
    def test_model_schema_missing_zero_and_unseen(self):
        text = "idX," + ",".join(ns.CATEGORICAL) + ",Effort-\n1," + ",".join([" a "]*22) + ",0\n2," + ",".join(["b"]*22) + ",10\n"
        _, rows, _, _ = ns.load(self.write(text)); model = ns.CategoricalNeighborModel(1).fit(rows)
        config = {n: "new" for n in ns.CATEGORICAL}; self.assertTrue(math.isfinite(model.predict_config(config)))
        config[ns.CATEGORICAL[0]] = ""; self.assertTrue(math.isfinite(model.predict_config(config)))
        with self.assertRaises(ValueError): model.predict_config({n: "x" for n in ns.CATEGORICAL[:-1]})
        self.assertEqual(model.predict(rows[0]), 0)
    def test_invalid_nonfinite_and_insufficient_groups(self):
        text = "idX," + ",".join(ns.CATEGORICAL) + ",Effort-\n1," + ",".join(["a"]*22) + ",NaN\n"
        _, rows, _, _ = ns.load(self.write(text))
        with self.assertRaises(ValueError): ns.split_groups(rows)
        with self.assertRaises(ValueError): ns.CategoricalNeighborModel().fit(rows)
if __name__ == "__main__": unittest.main()
