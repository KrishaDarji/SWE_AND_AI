import csv, os, tempfile, unittest
from source.analyzer import analyze, read_csv, direction, role

class AnalyzerTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_summaries_and_roles(self):
        r = analyze(self.fixture("idX,kind,Kloc+,Effort-,Defects-\n 1 , a , 2, 3, 4\n2,b,4,5,6\n"))
        self.assertEqual(r["record_count"], 2); self.assertEqual(r["outcomes"], ["Kloc+", "Effort-", "Defects-"])
        self.assertEqual(r["metadata"], ["idX"]); self.assertEqual(r["descriptors"], ["kind"])
        self.assertEqual(r["columns"][1]["frequencies"], {"a": 1, "b": 1})
        self.assertEqual(r["columns"][2]["statistics"]["mean"], 3)
    def test_missing_nonfinite_and_whitespace(self):
        r = analyze(self.fixture("x,y\n 1 , ?\nNaN, 2 \n, inf\n"))
        self.assertEqual(r["columns"][0]["statistics"]["count"], 1); self.assertEqual(r["columns"][0]["missing"], 2)
        self.assertEqual(r["columns"][1]["statistics"]["mean"], 2)
    def test_invalid_outcome_is_categorical(self):
        r = analyze(self.fixture("Kloc+\nnot-a-number\n")); self.assertEqual(r["columns"][0]["inferred_type"], "categorical")
    def test_empty_and_header_only(self):
        self.assertEqual(analyze(self.fixture(""))["record_count"], 0)
        self.assertEqual(analyze(self.fixture("a,b\n"))["record_count"], 0)
    def test_directions(self):
        self.assertEqual(direction("Kloc+"), "maximize"); self.assertEqual(direction("Effort-"), "minimize")
        self.assertEqual(role("MonthsX"), "metadata")

if __name__ == "__main__": unittest.main()
