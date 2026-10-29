import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.runs.treatment3.trials.trial1.nasa93dem.step04.source.project_flow import ProposalFlow


DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = ProposalFlow(DATA).fit()

    def test_fit_discovers_features_and_targets(self):
        self.assertGreater(len(self.flow.feature_columns), 10)
        self.assertEqual(set(self.flow.baseline), {"Effort-", "Defects-", "MonthsX"})

    def test_assessment_has_stable_contract(self):
        result = self.flow.assess({"name": "incoming", "Kloc+": 20, "rely": "h", "acap": "n"}, 100)
        self.assertEqual(result["name"], "incoming")
        self.assertIn(result["decision"], {"accept", "defer", "review"})
        self.assertGreaterEqual(result["risk_score"], 0)

    def test_unknown_categories_and_missing_fields_are_supported(self):
        result = self.flow.assess({"name": "sparse", "rely": "new-category"})
        self.assertIsInstance(result["predicted_effort"], float)


if __name__ == "__main__":
    unittest.main()
