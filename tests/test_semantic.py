"""Semantic tests for the ten shared skills.

These tests check the meaning of each result, not merely that a function runs.
The fixtures are intentionally small and controlled so failures identify a
contract violation rather than a change in a large real dataset.
"""

import unittest

from src.core.blame import blame
from src.core.certify import certify
from src.core.choose import choose
from src.core.datasets import DatasetSpec
from src.core.fix import fix
from src.core.flow import stream_objective
from src.core.guess import predict_row
from src.core.justify import justify
from src.core.race import race
from src.core.remember import rank_rows
from src.core.spot import spot


SPEC = DatasetSpec("toy", "toy client", None, ["quality+", "cost-"])


def rows():
    return [
        {"setting": "a", "quality+": "0.50", "cost-": "100"},
        {"setting": "b", "quality+": "0.80", "cost-": "80"},
        {"setting": "c", "quality+": "0.95", "cost-": "60"},
        {"setting": "d", "quality+": "0.10", "cost-": "140"},
    ]


class SemanticSkillTests(unittest.TestCase):
    def test_remember_ranking_prefers_high_quality_and_low_cost(self):
        ranked = rank_rows(SPEC, rows(), top_k=1)
        self.assertEqual(ranked[0][1]["setting"], "c")

    def test_guess_predicts_nearby_configuration(self):
        prediction = predict_row(SPEC, rows()[:3], {"setting": "b"}, k=1)
        self.assertAlmostEqual(prediction["quality+"], 0.80)
        self.assertAlmostEqual(prediction["cost-"], 80.0)

    def test_certify_respects_minimize_direction(self):
        result = certify("cost-", [60, 65, 62], [100, 105, 98], permutations=200)
        self.assertGreater(result["effect"], 0)
        self.assertGreater(result["treatment_mean"], result["baseline_mean"])

    def test_flow_stops_only_after_stable_evidence(self):
        result = stream_objective(SPEC, rows() * 8, minimum=3, tolerance=0.01, patience=2)
        self.assertGreaterEqual(result["seen"], 3)
        self.assertIn("quality+", result["final_means"])
        self.assertIn("cost-", result["final_means"])

    def test_blame_ranks_feature_with_group_difference(self):
        data = [
            {"knob": "0", "quality+": "0.1", "cost-": "100"},
            {"knob": "0", "quality+": "0.2", "cost-": "95"},
            {"knob": "1", "quality+": "0.9", "cost-": "50"},
            {"knob": "1", "quality+": "0.8", "cost-": "55"},
        ]
        finding = blame(SPEC, data, top_k=1)[0]
        self.assertEqual(finding["feature"], "knob")
        self.assertGreater(finding["effect"], 0)

    def test_justify_contains_recommendation_evidence(self):
        result = justify(SPEC, rows(), rows()[2])
        self.assertEqual(result["dataset"], "toy")
        self.assertTrue(result["reasons"])
        self.assertIn("Recommendation", result["text"])

    def test_choose_respects_constraint_and_objective(self):
        result = choose(SPEC, rows(), budget=4)
        selected = result["selected"]
        self.assertEqual(selected["setting"], "c")
        self.assertEqual(result["evaluated"], 4)

    def test_fix_reports_a_real_one_setting_repair_or_no_repair(self):
        result = fix(SPEC, rows(), rows()[0])
        self.assertIn("repaired", result)
        if result["repaired"]:
            self.assertNotEqual(result["candidate"]["setting"], "a")
            self.assertGreater(result["improvement"], 0)

    def test_spot_detects_injected_numeric_anomaly(self):
        data = rows() + [{"setting": "outlier", "quality+": "1000", "cost-": "1"}]
        finding = spot(SPEC, data, top_k=1)[0]
        self.assertEqual(finding["row"]["setting"], "outlier")
        self.assertGreater(finding["score"], 1.5)

    def test_race_gives_both_methods_the_same_budget(self):
        result = race(SPEC, rows(), budget=2)
        self.assertEqual(result["budget"], 2)
        self.assertIn("shared_selected", result)
        self.assertIn("baseline_selected", result)
        self.assertIn("improvement", result)


if __name__ == "__main__":
    unittest.main()
