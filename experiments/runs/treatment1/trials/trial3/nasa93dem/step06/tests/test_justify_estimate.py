import csv, importlib.util, tempfile, unittest
from pathlib import Path
SOURCE = Path(__file__).parents[1] / "source" / "justify_estimate.py"
SPEC = importlib.util.spec_from_file_location("justify_estimate", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)

class EstimateTests(unittest.TestCase):
    def test_nearest_peer_median(self):
        records = [(10,100,10),(11,110,11),(12,120,12),(100,999,99),(9,90,9)]
        result = MODULE.estimate(records, 11)
        self.assertEqual(result["peer_count"], 5); self.assertEqual(result["effort_median"], 110)
    def test_bad_csv_rows_are_skipped(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.writer(handle); writer.writerow(["Kloc+","Effort-","MonthsX"])
            writer.writerow(["10","100","10"]); writer.writerow(["bad","100","10"]); path = handle.name
        self.assertEqual(MODULE.read_records(path), [(10.0,100.0,10.0)])
    def test_render_has_sponsor_case(self):
        text = MODULE.render(MODULE.estimate([(10,100,10)] * 5, 10))
        self.assertIn("planning range", text); self.assertIn("Sponsor decision", text)

if __name__ == "__main__": unittest.main()

