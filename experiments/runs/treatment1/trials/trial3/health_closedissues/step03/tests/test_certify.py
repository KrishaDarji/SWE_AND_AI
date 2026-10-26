import csv, pathlib, sys, tempfile, unittest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import certify

class CertifyTests(unittest.TestCase):
    def test_score_direction(self):
        self.assertAlmostEqual(certify.score({"MRE-":".2", "ACC+":".8", "PRED40+":".5"}), 1.1)

    def test_winner_on_clear_synthetic_data(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as f:
            w = csv.DictWriter(f, fieldnames=["MRE-", "ACC+", "PRED40+", "criterion"])
            w.writeheader()
            for _ in range(30): w.writerow({"MRE-":".1", "ACC+":".9", "PRED40+":".9", "criterion":"best"})
            for g in ("other1", "other2"):
                for _ in range(30): w.writerow({"MRE-":".8", "ACC+":".2", "PRED40+":".1", "criterion":g})
            f.flush()
            self.assertEqual(certify.certify(f.name, samples=200), 0)

if __name__ == "__main__": unittest.main()
