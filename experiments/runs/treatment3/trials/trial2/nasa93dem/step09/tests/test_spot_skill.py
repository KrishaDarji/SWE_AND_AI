import csv, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import spot_skill

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/process/nasa93dem.csv"))

class SpotTests(unittest.TestCase):
    def test_schema_and_count(self):
        rows = spot_skill.load_validate(DATA)
        self.assertEqual(len(rows), 93)
        self.assertEqual(rows[0]["idX"], "1")
    def test_reproducible_result(self):
        a = spot_skill.analyze(DATA); b = spot_skill.analyze(DATA)
        self.assertEqual(a[2], b[2]); self.assertEqual(a[1].tolist(), b[1].tolist())
    def test_explanation_present(self):
        rows, scores, top, details = spot_skill.analyze(DATA)
        self.assertTrue(0 <= top < len(rows)); self.assertGreater(scores[top], 0); self.assertGreaterEqual(len(details), 3)

if __name__ == "__main__": unittest.main()
