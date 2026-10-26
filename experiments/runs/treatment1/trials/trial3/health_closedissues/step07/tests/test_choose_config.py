import csv, tempfile, unittest, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import choose_config

class ChooseTests(unittest.TestCase):
    def make(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        w = csv.DictWriter(f, fieldnames=["MRE-", "ACC+", "PRED40+", "N_estimators"])
        w.writeheader(); w.writerows(rows); f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_balanced_candidate_wins(self):
        p = self.make([{"MRE-":"0", "ACC+":"1", "PRED40+":"0", "N_estimators":"1"}, {"MRE-":".2", "ACC+":".8", "PRED40+":".8", "N_estimators":"2"}])
        result = choose_config.choose(p)
        self.assertEqual(result["configuration"]["N_estimators"], "2")
        self.assertAlmostEqual(result["score"], .8)

    def test_invalid_and_empty_inputs_fail(self):
        p = self.make([{"MRE-":"1.2", "ACC+":".5", "PRED40+":".5", "N_estimators":"1"}])
        with self.assertRaises(ValueError): choose_config.choose(p)
        self.assertRaises(ValueError, choose_config.choose, self.make([]))

if __name__ == "__main__":
    unittest.main()
