import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
import guess_model


class GuessModelTests(unittest.TestCase):
  def test_read_data_schema_and_numeric_targets(self):
    path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv")
    x, y = guess_model.read_data(path)
    self.assertEqual(len(x), len(y))
    self.assertGreater(len(x), 10)
    self.assertEqual(len(x[0]), len(guess_model.FEATURES))
    self.assertEqual(len(y[0]), 3)


  def test_model_runs_and_reports_all_targets(self):
    path = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv")
    result = guess_model.run(path)
    self.assertEqual(set(result["targets"]), set(guess_model.TARGETS))
    self.assertTrue(all("mae" in result["targets"][name] for name in guess_model.TARGETS))


  def test_unknown_category_is_supported(self):
    model = guess_model.build_model()
    model.fit([["1", "known", "1", "1", "1"]], [[0.2, 0.8, 0.5]])
    model.predict([["1", "new", "1", "1", "1"]])


if __name__ == "__main__":
    unittest.main()
