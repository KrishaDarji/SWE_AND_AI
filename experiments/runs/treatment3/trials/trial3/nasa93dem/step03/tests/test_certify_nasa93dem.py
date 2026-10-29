import csv, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import certify_nasa93dem as c

class CertifyTests(unittest.TestCase):
    def test_dataset_and_result_contract(self):
        p = Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv"
        r = c.certify(p)
        self.assertEqual(r["n_total"], 93); self.assertEqual(r["groups"], 5)
        self.assertGreater(r["n_low"], 0); self.assertGreater(r["n_high"], 0)
        self.assertGreaterEqual(r["ci_high"], r["ci_low"])
        self.assertGreaterEqual(r["p"], 0); self.assertLessEqual(r["p"], 1)
    def test_validation_rejects_missing_columns(self):
        with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv") as f:
            csv.writer(f).writerow(["x"]); f.flush()
            with self.assertRaises(ValueError): c.load(f.name)

if __name__ == "__main__": unittest.main()
