import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nasa93_summary", ROOT / "source" / "nasa93_summary.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class Step4Tests(unittest.TestCase):
    def test_step1_summary_is_preserved(self):
        result = module.summarize_projects(DATA)
        self.assertEqual(93, result["project_count"])
        self.assertEqual("1", module.load_projects(DATA)[0]["id"])
        json.dumps(result)

    def test_step2_estimate_is_preserved(self):
        result = module.estimate_effort(DATA, 10)
        self.assertEqual("needs_budget_review", result["approval"])
        self.assertLess(result["planning_low"], result["planning_high"])

    def test_step3_comparison_is_preserved(self):
        result = module.compare_costs(DATA, "centerX", "1", "2", samples=100)
        self.assertEqual(12, result["count_a"])
        self.assertIsInstance(result["genuinely_less_costly"], bool)

    def test_assesses_approved_arriving_proposal(self):
        estimate = module.estimate_effort(DATA, 10)
        result = module.assess_proposal(DATA, {"id": "P-1", "kloc": 10,
                                                "budget": estimate["planning_high"],
                                                "max_months": 100})
        self.assertEqual("P-1", result["proposal_id"])
        self.assertEqual("approve", result["decision"])
        self.assertTrue(result["constraints_checked"]["budget"])

    def test_rejects_proposal_that_breaks_budget(self):
        result = module.assess_proposal(DATA, {"id": "P-2", "kloc": 10, "budget": 0})
        self.assertEqual("reject", result["decision"])

    def test_review_without_constraints_and_batch_preserves_arrival_order(self):
        results = module.assess_proposals(DATA, [{"id": "first", "kloc": 1},
                                                   {"id": "second", "kloc": 2}])
        self.assertEqual(["first", "second"], [item["proposal_id"] for item in results])
        self.assertTrue(all(item["decision"] == "needs_review" for item in results))

    def test_proposal_validation(self):
        with self.assertRaises(ValueError):
            module.assess_proposal(DATA, {"id": "bad", "kloc": 0})
        with self.assertRaises(ValueError):
            module.assess_proposals(DATA, {"id": "bad", "kloc": 1})

    def test_step5_ranks_factors_associated_with_excessive_effort(self):
        result = module.analyze_effort_factors(DATA)
        self.assertEqual(93, result["project_count"])
        self.assertEqual("effort_per_kloc above dataset median", result["definition"])
        self.assertTrue(result["factors"])
        self.assertIn("factor", result["factors"][0])
        self.assertIn("levels", result["factors"][0])
        self.assertTrue(all(level["count"] >= 3
                            for factor in result["factors"]
                            for level in factor["levels"]))

    def test_step5_is_deterministic_and_validates_level_count(self):
        self.assertEqual(module.analyze_effort_factors(DATA),
                         module.analyze_effort_factors(DATA))
        with self.assertRaises(ValueError):
            module.analyze_effort_factors(DATA, 0)

    def test_step6_justifies_estimate_with_evidence_and_limitations(self):
        result = module.justify_estimate(DATA, 10)
        self.assertEqual("needs_budget_review", result["recommendation"])
        self.assertEqual(93, result["evidence"]["historical_projects"])
        self.assertIn("median effort", result["sponsor_message"])
        self.assertEqual(3, len(result["risks_and_limitations"]))
        self.assertEqual(module.estimate_effort(DATA, 10), result["estimate"])

    def test_step6_budget_justification_changes_recommendation(self):
        estimate = module.estimate_effort(DATA, 10)
        result = module.justify_estimate(DATA, 10, estimate["planning_high"])
        self.assertEqual("approve", result["recommendation"])

    def test_step6_validates_estimate_inputs(self):
        with self.assertRaises(ValueError):
            module.justify_estimate(DATA, 0)

    def test_step7_selects_highest_value_within_budget(self):
        result = module.choose_project_settings(DATA, 100)
        self.assertEqual("select", result["decision"])
        self.assertGreater(result["selected"]["Kloc+"], 0)
        self.assertLessEqual(result["selected"]["Effort-"], 100)
        self.assertGreater(result["candidate_count"], 0)

    def test_step7_applies_schedule_and_quality_constraints(self):
        result = module.choose_project_settings(DATA, 100000, max_months=10,
                                                max_defects=10)
        self.assertTrue(result["candidate_count"] <= 93)
        if result["selected"]:
            self.assertLessEqual(result["selected"]["MonthsX"], 10)
            self.assertLessEqual(result["selected"]["Defects-"], 10)

    def test_step7_reports_no_feasible_setting_and_validates_budget(self):
        result = module.choose_project_settings(DATA, 0)
        self.assertEqual("no_feasible_setting", result["decision"])
        self.assertIsNone(result["selected"])
        with self.assertRaises(ValueError):
            module.choose_project_settings(DATA, -1)


if __name__ == "__main__":
    unittest.main()
