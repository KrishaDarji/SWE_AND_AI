import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[8]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"source"))
import certify

class CertifyTests(unittest.TestCase):
    def test_load_and_schema(self):
        rows=certify.load_validate(ROOT/"moot/optimize/hpo/Health-ClosedIssues0000.csv")
        self.assertEqual(len(rows),10000)
        self.assertTrue(set(certify.TARGETS)<=set(rows[0]))
    def test_reproducible_statistics(self):
        rows=certify.load_validate(ROOT/"moot/optimize/hpo/Health-ClosedIssues0000.csv")
        x=certify.analyze(rows); y=certify.analyze(rows)
        self.assertEqual(x,y)
        self.assertGreater(x["n_a"],0); self.assertGreater(x["n_b"],0)
        self.assertGreaterEqual(x["p_value"],0); self.assertLessEqual(x["p_value"],1)

if __name__ == "__main__": unittest.main()
