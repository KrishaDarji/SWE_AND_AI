import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow_skill import evaluate, parse_proposal, run  # noqa: E402


class FlowSkillTests(unittest.TestCase):
    def test_dataset_has_sequential_valid_records(self):
        result = run(Path(__file__).parents[8] / "moot/optimize/process/nasa93dem.csv")
        self.assertEqual(result["records_seen"], 93)
        self.assertEqual(result["valid_records"], 93)
        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["best"]["proposal_id"], "58")

    def test_invalid_and_tie_are_deterministic(self):
        text = "idX,Effort-,Defects-,MonthsX\n2,10,10,10\n1,10,10,10\nX,,2,3\n"
        state, errors = evaluate(io.StringIO(text))
        self.assertEqual(state.error_count, 1)
        self.assertEqual(state.best.proposal_id, "1")
        self.assertEqual(errors, ["line 4: missing Effort-"])

    def test_negative_and_nonfinite_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_proposal({"idX": "1", "Effort-": "-1", "Defects-": "1", "MonthsX": "1"})


if __name__ == "__main__":
    unittest.main()
