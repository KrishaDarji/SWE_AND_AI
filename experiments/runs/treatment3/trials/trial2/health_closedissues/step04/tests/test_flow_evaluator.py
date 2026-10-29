import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow_evaluator import EvaluationState, evaluate_rows

BASE = {"MRE-":"0.2", "ACC+":"0.8", "PRED40+":"0.7", "N_estimators":"10",
        "criterion":"x", "Min_sample_leaves":"1", "Min_impurity_decrease":"0.1", "Max_depth":"3"}

class FlowTests(unittest.TestCase):
  def test_sequential_best_and_running_mean(self):
    a = dict(BASE); b = dict(BASE, **{"MRE-":"0.1", "ACC+":"0.1"})
    state = evaluate_rows([a, b])
    self.assertEqual((state.valid, state.invalid), (2, 0))
    self.assertEqual(state.best["MRE-"], "0.1")
    for metric, expected in {"MRE-": 0.15, "ACC+": 0.45, "PRED40+": 0.7}.items():
      self.assertAlmostEqual(state.running[metric], expected)

  def test_tie_uses_stable_configuration_order(self):
    a = dict(BASE, N_estimators="20"); b = dict(BASE, N_estimators="10")
    self.assertEqual(evaluate_rows([a, b]).best["N_estimators"], "10")

  def test_invalid_and_incomplete_are_counted(self):
    bad = dict(BASE, **{"ACC+":"nan"}); incomplete = dict(BASE); del incomplete["Max_depth"]
    state = evaluate_rows([bad, incomplete, BASE])
    self.assertEqual((state.observations, state.valid, state.invalid), (3, 1, 2))

  def test_empty_is_deterministic(self):
    state = EvaluationState()
    self.assertIsNone(state.best)
    self.assertEqual(state.running, {"MRE-": 0.0, "ACC+": 0.0, "PRED40+": 0.0})

if __name__ == "__main__":
    unittest.main()
