import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from flow import ProposalFlow, read_proposals  # noqa: E402


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flow = ProposalFlow.from_csv(Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv")

    def test_assessment_has_expected_contract(self):
        proposal = {"name": "small", "centerX": "2", "YearX": "1979", "prec": "h", "flex": "h", "resl": "h"}
        result = self.flow.assess(proposal)
        self.assertIn("estimates", result)
        self.assertEqual(set(result["estimates"]), {"Kloc+", "Effort-", "Defects-", "MonthsX"})
        self.assertIn(result["confidence"], {"high", "medium", "low"})

    def test_stream_accepts_json_like_records(self):
        results = list(self.flow.assess_stream([{"id": "p1", "prec": "h"}, {"id": "p2", "prec": "l"}]))
        self.assertEqual([r["proposal"] for r in results], ["p1", "p2"])

    def test_missing_history_is_rejected(self):
        with self.assertRaises(ValueError):
            ProposalFlow([])


if __name__ == "__main__":
    unittest.main()
