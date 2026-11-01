import csv, math, os, tempfile, unittest, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from nasa93_summary import read_dataset, summarize, EffortGuesser, grouped_split, guess

class SummaryTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_summaries_roles_and_directions(self):
        p = self.fixture("idX,kind,feature,Kloc+,Effort-,Defects-,bad\n1,a, 2 ,10,20,3,\n2,a,4,20,40,?,oops\n3,b,?,30,inf,5,7\n")
        s = summarize(p)
        self.assertEqual(s["record_count"], 3); self.assertEqual(s["categorical"]["kind"], {"a": 2, "b": 1})
        self.assertEqual(s["numeric"]["feature"]["count"], 2); self.assertEqual(s["missing"]["feature"], 1)
        self.assertEqual(s["invalid_numeric"]["bad"], 0); self.assertEqual(s["roles"]["idX"], "metadata")
        self.assertEqual(s["outcomes"]["Kloc+"]["direction"], "maximize")
        self.assertEqual(s["outcomes"]["Effort-"]["direction"], "minimize")
        self.assertNotIn("Kloc+", s["model_predictors"])
    def test_whitespace_invalid_and_header_only(self):
        p = self.fixture("A,b\n 1 , x \n nan, y\n2,\n")
        h, rows, _ = read_dataset(p); self.assertEqual(rows[0]["A"], "1"); self.assertEqual(summarize(p)["invalid_numeric"]["A"], 0)
        q = self.fixture("A,b\n"); self.assertEqual(summarize(q)["record_count"], 0)
        r = self.fixture(""); self.assertEqual(summarize(r)["columns"], [])
    def test_known_statistics(self):
        p = self.fixture("A,b\n1,x\n3,x\n5,y\n")
        self.assertEqual(summarize(p)["numeric"]["A"]["mean"], 3.0)
        self.assertEqual(summarize(p)["numeric"]["A"]["median"], 3.0)

class GuessTests(unittest.TestCase):
    def fixture(self, text):
        f=tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def rows(self):
        return [{"a": " x ", "b": "h", "Effort-": "10"}, {"a": "x", "b": "h", "Effort-": "20"},
                {"a": "y", "b": "l", "Effort-": "30"}, {"a": "z", "b": "l", "Effort-": "40"}]
    def test_grouped_reproducible_and_training_only(self):
        a,b=grouped_split(self.rows(), ["a","b"]); c,d=grouped_split(self.rows(), ["a","b"])
        self.assertEqual(a,c); self.assertEqual(b,d); self.assertTrue(set(map(id,a)).isdisjoint(map(id,b)))
        m=EffortGuesser(["a","b"]).fit(a); self.assertEqual(m.mean, statistics.fmean([float(r["Effort-"]) for r in a]))
    def test_missing_unseen_and_whitespace(self):
        m=EffortGuesser(["a","b"]).fit(self.rows()[:2]); self.assertEqual(m.predict([{ "a":"x","b":"h"}])[0],15)
        self.assertEqual(m.predict([{ "a":"new","b":"q"}])[0],15)
    def test_excluded_columns_and_proposal_validation(self):
        p=self.fixture("idX,a,Kloc+,Effort-,Defects-\n1,x,1,10,2\n2,y,2,20,3\n3,z,3,30,4\n4,w,4,40,5\n")
        out=guess(p); self.assertEqual(out["predictors"],["a"]); self.assertIn("baseline_mae",out)
        with self.assertRaises(ValueError): guess(p,{"wrong":"x"})
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): grouped_split([{ "a":"x"},{"a":"x"}], ["a"])

if __name__ == "__main__": unittest.main()
