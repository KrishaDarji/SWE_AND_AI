import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class ChooseTests(unittest.TestCase):
    def rows(self):
        return [
            {"a":"0", "b":"0", "Performance-":"bad"},
            {"a":"1.0", "b":"0", "Performance-":"2"},
            {"a":"1", "b":"0", "Performance-":"1"},
            {"a":"0", "b":"1", "Performance-":"2"},
            {"a":"1", "b":"1", "Performance-":"3"},
        ]

    def test_budget_duplicates_winner_and_full_settings(self):
        r = summary.choose_rows(self.rows(), 20, 42)
        self.assertLessEqual(r["attempted_evaluations"], 20)
        self.assertEqual(r["distinct_configurations"], 4)
        self.assertEqual(r["winner"], {"position": 2, "settings": {"a": 1, "b": 0}, "value": 2.0})
        self.assertEqual(len({tuple(x["configuration"]) for x in r["evaluation_history"]}), r["attempted_evaluations"])

    def test_reproducible_and_invalid_attempt_counts(self):
        a = summary.choose_rows(self.rows(), 2, 42, audit=False)
        b = summary.choose_rows(self.rows(), 2, 42, audit=False)
        self.assertEqual(a, b)
        self.assertEqual(a["attempted_evaluations"], 2)
        self.assertLessEqual(a["successful_evaluations"], 2)

    def test_budget_edges_and_empty(self):
        self.assertEqual(summary.choose_rows([], 0)["status"], "zero_budget")
        self.assertEqual(summary.choose_rows(self.rows(), 0)["winner"], None)
        self.assertEqual(summary.choose_rows([{"a":"0", "Performance-":"?"}], 1)["status"], "no_successful_evaluations")
        with self.assertRaises(ValueError): summary.choose_rows(self.rows(), -1)
        with self.assertRaises(ValueError): summary.choose_rows(self.rows(), 1.5)

    def test_audit_is_separate_and_cannot_change_winner(self):
        base = summary.choose_rows(self.rows(), 1, 42, audit=False)
        audited = summary.choose_rows(self.rows(), 1, 42, audit=True)
        self.assertEqual(base["winner"], audited["winner"])
        self.assertIn("retrospective_historical_replay", audited)
        self.assertIn("not newly measured encoding", audited["retrospective_historical_replay"]["note"])

if __name__ == "__main__": unittest.main()
