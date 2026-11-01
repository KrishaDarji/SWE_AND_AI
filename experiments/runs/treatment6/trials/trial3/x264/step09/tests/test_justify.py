import math
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class JustifyTests(unittest.TestCase):
    def test_winner_earliest_tie_and_complete_configuration(self):
        rows = [
            {"a":"0", "b":"1", "Performance-":"5"},
            {"a":"1", "b":"0", "Performance-":"2"},
            {"a":"0", "b":"1", "Performance-":"2"},
        ]
        result = summary.justify_rows(rows)
        self.assertEqual(result["recommendation"]["position"], 2)
        self.assertEqual(result["recommendation"]["settings"], {"a":1, "b":0})
        self.assertEqual(result["recommendation"]["value"], 2.0)
        self.assertEqual(result["reference"], {"count":3, "median":2.0})

    def test_exclusions_keep_original_positions_and_invalid_outcomes(self):
        rows = [{"a":"0", "b":"0", "Performance-":"?"},
                {"a":"2", "b":"0", "Performance-":"1"},
                {"a":"1", "b":"1", "Performance-":"nan"},
                {"a":"1", "b":"0", "Performance-":"3"}]
        result = summary.justify_rows(rows)
        self.assertEqual(result["recommendation"]["position"], 4)
        self.assertEqual([x["position"] for x in result["excluded"]], [1, 2, 3])

    def test_medians_reductions_and_zero_reference(self):
        rows = [{"a":"0", "Performance-":str(v)} for v in (1, 9, 3, 7)]
        result = summary.justify_rows(rows)
        self.assertEqual(result["reference"]["median"], 5.0)
        self.assertEqual(result["comparison"]["absolute_reduction"], 4.0)
        self.assertEqual(result["comparison"]["percentage_reduction"], 80.0)
        zero = summary.justify_rows([{"a":"0", "Performance-":"0"}, {"a":"1", "Performance-":"0"}])
        self.assertIsNone(zero["comparison"]["percentage_reduction"])

    def test_empty_and_no_eligible(self):
        self.assertEqual(summary.justify_rows([])["status"], "empty_input")
        result = summary.justify_rows([{"a":"2", "Performance-":"1"}])
        self.assertEqual(result["status"], "no_eligible_recommendation")
        self.assertIsNone(result["recommendation"])

    def test_claim_boundaries_and_global_association(self):
        result = summary.justify_rows([{"a":"0", "Performance-":"1"}, {"a":"1", "Performance-":"2"}])
        self.assertIn("retrospective", result["selection_basis"])
        assumptions = result["associations"]["assumptions"]
        self.assertTrue(any("not causal" in x for x in assumptions))
        self.assertNotIn("local", result["selection_basis"].lower())

if __name__ == "__main__": unittest.main()
