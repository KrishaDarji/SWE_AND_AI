import csv, math, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summary import parse_cell, summarize, format_report, OUTCOMES

class SummaryTests(unittest.TestCase):
    def make(self, text):
        fixture_dir = Path(__file__).parents[1] / "fixtures"
        fixture_dir.mkdir(exist_ok=True)
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", dir=fixture_dir, delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True)); return f.name
    def test_numeric_and_categories(self):
        s = summarize(self.make("A,b,Kloc+,Effort-,Defects-,metaX\n1, x,2,4,8,z\n3,x,6,8,12,z\n"))
        self.assertEqual(s["record_count"], 2); self.assertEqual(s["numeric"]["A"]["mean"], 2)
        self.assertEqual(s["categorical"]["b"], {"x": 2}); self.assertEqual(s["roles"]["metadata"], ["metaX"])
    def test_whitespace_missing_nonfinite_and_invalid_outcome(self):
        self.assertEqual(parse_cell("  2.5 "), 2.5); self.assertIsNone(parse_cell(" NaN ")); self.assertIsNone(parse_cell(" ? "))
        s = summarize(self.make("Kloc+,c\n 2 , a\ninf,b\n")); self.assertEqual(s["missing"]["Kloc+"], 1)
        with self.assertRaises(ValueError): summarize(self.make("Kloc+,c\nnot-a-number,a\n"))
    def test_roles_and_directions(self):
        s = summarize(self.make("idX,Kloc+,Effort-,Defects-,kind\n1,2,3,4,h\n"))
        self.assertEqual(s["roles"]["metadata"], ["idX"]); self.assertEqual(s["roles"]["outcomes"]["Effort-"]["direction"], "minimize"); self.assertIn("kind", s["roles"]["descriptors"])
        self.assertEqual(OUTCOMES["Kloc+"], "maximize")
    def test_empty_and_header_only(self):
        empty = summarize(self.make("")); report = format_report(empty)
        self.assertEqual(empty["roles"], {"descriptors": [], "metadata": [], "outcomes": {}})
        self.assertIn("Records: 0", report); self.assertIn("Roles: descriptors=; metadata=", report); self.assertIn("Outcomes: ", report)
        head = summarize(self.make("a,b\n")); report = format_report(head)
        self.assertEqual(head["columns"], ["a","b"]); self.assertEqual(head["missing"], {"a":0,"b":0}); self.assertIn("a: unknown", report)
    def test_normal_data_complete_report(self):
        report = format_report(summarize(self.make("idX,Kloc+,Effort-,Defects-,kind\n1,2,3,4,h\n")))
        self.assertIn("Records: 1", report); self.assertIn("Kloc+ (maximize)", report); self.assertIn("kind: h=1", report)
    def test_bad_shape(self):
        with self.assertRaises(ValueError): summarize(self.make("a,b\n1\n"))

if __name__ == "__main__": unittest.main()
