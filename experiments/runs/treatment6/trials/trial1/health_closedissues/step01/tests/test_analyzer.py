import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyzer import summarize, CONFIG, OUTCOMES, DIRECTIONS

HEADER = ",".join(OUTCOMES + CONFIG)
class AnalyzerTests(unittest.TestCase):
    def run_case(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return summarize(f.name)
    def test_complete_reporting_path(self):
        r = self.run_case(HEADER + "\n1,0.5,1,10,squared_error,2,0.1,4\n 2 , NaN , ? ,10,squared_error,2,0.1,4\n3,0.7,0,20,absolute_error,3,0.2,5\n")
        self.assertEqual(r["record_count"], 3); self.assertEqual(r["column_details"]["MRE-"]["statistics"]["mean"], 2)
        self.assertEqual(r["column_details"]["ACC+"]["missing"], 1); self.assertEqual(r["column_details"]["criterion"]["frequencies"], {"absolute_error": 1, "squared_error": 2})
        self.assertEqual(r["configuration_counts"]["distinct"], 2); self.assertEqual(r["configuration_counts"]["duplicate_rows"], 1); self.assertEqual(r["configuration_counts"]["duplicate_rows_exact"], 0)
        self.assertEqual(r["optimization_directions"], DIRECTIONS); self.assertEqual(r["column_details"]["criterion"]["type"], "categorical")
    def test_empty_and_header_only(self):
        self.assertEqual(summarize(self.path(""))["record_count"], 0)
        self.assertTrue(summarize(self.path(HEADER + "\n"))["empty"])
    def path(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", delete=False); f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
if __name__ == "__main__": unittest.main()
