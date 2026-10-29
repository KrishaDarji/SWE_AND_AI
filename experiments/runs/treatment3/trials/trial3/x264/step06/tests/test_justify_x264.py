import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify_x264 import CONFIG, DATASET, analyze, load_validate, render


class JustifyTests(unittest.TestCase):
  def test_load_and_validate(self):
    rows = load_validate(DATASET)
    self.assertEqual(len(rows), 1152)
    self.assertEqual(len({tuple(bits) for bits, score in rows}), 1152)


  def test_recommendation_is_observed_and_maximal(self):
    rows, best, score, *_ = analyze()
    self.assertEqual(score, max(value for bits, value in rows))
    self.assertIn((best, score), rows)


  def test_report_mentions_limits_and_uncertainty(self):
    report = render()
    self.assertIn("Bootstrap 95%", report)
    self.assertIn("historical association", report)
    self.assertIn("quality metrics", report)


if __name__ == "__main__":
  unittest.main()
