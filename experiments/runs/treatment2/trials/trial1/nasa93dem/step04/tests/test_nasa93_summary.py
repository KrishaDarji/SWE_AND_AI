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


if __name__ == "__main__":
    unittest.main()
