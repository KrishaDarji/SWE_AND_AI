import csv, math, os, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from nasa93_summary import read_dataset, summarize

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

if __name__ == "__main__": unittest.main()
