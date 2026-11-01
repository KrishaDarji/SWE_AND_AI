import csv, math, pathlib, tempfile, unittest
import sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa_summary as n

class TestSummary(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: pathlib.Path(f.name).unlink(missing_ok=True)); return f.name
    def test_roles_frequencies_summaries(self):
        p = self.write("idX,prec,Kloc+,Effort-,Defects-\nA,h,1,2,3\nA,l,3,4,5\n")
        d=n.load(p); s=n.summarize(d)
        self.assertEqual(s["record_count"],2); self.assertEqual(s["roles"]["prec"],"categorical"); self.assertEqual(s["categorical_frequencies"]["prec"],{"h":1,"l":1}); self.assertEqual(s["duplicate_idX"],["A"]); self.assertEqual([x["position"] for x in d["projects"]],[2,3]); self.assertEqual(s["outcomes"]["Kloc+"]["median"],2); self.assertEqual(s["outcomes"]["Kloc+"]["sample_variance"],2)
    def test_missing_nonfinite_invalid_and_all_missing(self):
        p=self.write("Kloc+,Effort-,Defects-,prec\n?,bad,NaN,h\n,inf,?,l\n")
        s=n.summarize(n.load(p)); self.assertEqual(s["outcomes"]["Kloc+"]["count"],0); self.assertEqual(s["invalid_numeric"]["Effort-"],2); self.assertEqual(s["invalid_numeric"]["Defects-"],1); self.assertEqual(s["missing"]["Defects-"],1); self.assertEqual(s["categorical_frequencies"]["prec"],{"h":1,"l":1})
    def test_empty_header_malformed_blank(self):
        e=self.write(""); self.assertEqual(n.load(e)["record_count"] if "record_count" in n.load(e) else len(n.load(e)["projects"]),0)
        h=self.write("a,b\n"); self.assertEqual(len(n.load(h)["projects"]),0)
        m=self.write("a,b\n1\n\n2,3\n"); d=n.load(m); self.assertEqual(len(d["projects"]),1); self.assertEqual(d["blank_rows"],1); self.assertEqual(len(d["errors"]),1)
    def test_median_even_odd(self): self.assertEqual(n.median([3,1,2]),2); self.assertEqual(n.median([4,1,3,2]),2.5)

class TestGuess(unittest.TestCase):
    def row(self, i, value, token="h"):
        return {"values": {"idX": str(i), "Effort-": str(value), **{c: token for c in n.PREDICTORS}}}
    def test_schema_zero_and_mean(self):
        model=n.CategoricalKNN([self.row(1,0), self.row(2,10)])
        cfg={c:"h" for c in n.PREDICTORS}; self.assertEqual(model.predict(cfg),5)
        with self.assertRaises(ValueError): model.predict({c:"h" for c in n.PREDICTORS[:-1]})
        with self.assertRaises(ValueError): model.predict({**cfg,"extra":"x"})
    def test_group_split_and_reproducible(self):
        rows=[self.row(i,i,"h" if i<2 else "l") for i in range(4)]
        a,b=n.split_groups(rows); c,d=n.split_groups(rows)
        self.assertEqual([x["values"]["idX"] for x in a],[x["values"]["idX"] for x in c])
        self.assertTrue(set(tuple(x["values"][k] for k in n.PREDICTORS) for x in a).isdisjoint(set(tuple(x["values"][k] for k in n.PREDICTORS) for x in b)))
    def test_nonfinite_excluded_and_whitespace(self):
        r=self.row(1,0," h "); r["values"]["Effort-"]="NaN"
        self.assertEqual(n._cell("  "),"?"); self.assertIsNone(n._effort("inf"))
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): n.split_groups([self.row(1,1)])

class TestCertify(unittest.TestCase):
    def test_welch_direction_variance_and_threshold(self):
        x=n.welch_test([10,12,14],[7,8,9],1)
        self.assertEqual(x["difference"],-4); self.assertAlmostEqual(x["standard_error"],math.sqrt(4/3+1/3)); self.assertLess(x["df"],10); self.assertEqual(x["threshold"]["point_meets"],True)
    def test_t_reference_values_and_invalid_threshold(self):
        self.assertAlmostEqual(n.t_cdf(0,1),.5); self.assertAlmostEqual(n.t_cdf(1.9599639845,100),.9736,places=3)
        with self.assertRaises(ValueError): n.welch_test([1,2],[1,2],-1)
    def test_edge_cases(self):
        self.assertEqual(n.welch_test([1],[2])["status"],"insufficient samples")
        self.assertEqual(n.welch_test([1,1],[2,2])["status"],"zero standard error")
if __name__ == "__main__": unittest.main()
