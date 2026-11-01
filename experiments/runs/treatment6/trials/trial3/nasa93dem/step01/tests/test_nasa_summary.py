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
if __name__ == "__main__": unittest.main()
