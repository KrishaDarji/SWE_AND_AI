import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../source"))
from justify import load_data, main
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/process/nasa93dem.csv"))
class JustifyTests(unittest.TestCase):
    def test_validation_and_count(self): self.assertEqual(len(load_data(DATA)), 93)
    def test_report_and_metrics(self):
        report, m = main(DATA); self.assertIn("Causality", report); self.assertGreater(m["estimate"], 0); self.assertGreater(m["mae"], 0)
if __name__ == "__main__": unittest.main()
