import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import certify_nasa93dem as cert


DATA = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"


class CertifyTests(unittest.TestCase):
  def test_load_and_schema(self):
    rows = cert.load_validate(DATA)
    self.assertEqual(len(rows), 93)
    self.assertIn(cert.TARGET, rows[0])
    self.assertIn(cert.GROUP, rows[0])


  def test_analysis_is_reproducible_and_has_required_statistics(self):
    rows = cert.load_validate(DATA)
    first, second = cert.analyze(rows), cert.analyze(rows)
    self.assertEqual((len(first["a"]), len(first["b"])), (37, 39))
    self.assertGreater(first["observed"], 0)
    self.assertEqual(first["ci"], second["ci"])
    self.assertEqual(first["p_value"], second["p_value"])
    self.assertTrue(0 <= first["p_value"] <= 1)


  def test_report_mentions_noncausal_limitation(self):
    report = cert.format_report(cert.analyze(cert.load_validate(DATA)))
    self.assertIn("not proof", report)
    self.assertIn("permutation p-value", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
