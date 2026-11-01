import unittest
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "source"))
import summary

class FixTests(unittest.TestCase):
    def test_minimum_change_then_value_then_position_and_normalization(self):
        rows = [
            {"a":"0", "b":"0.0", "Performance-":"100"},
            {"a":"1.0", "b":"1", "Performance-":"1"},
            {"a":"0", "b":"1", "Performance-":"90"},
            {"a":"1", "b":"0.0", "Performance-":"90"},
        ]
        result = summary.fix_rows(rows)
        self.assertEqual(result["recommendation"]["position"], 3)
        self.assertEqual(result["recommendation"]["changed_field_count"], 1)
        self.assertEqual(result["recommendation"]["changed_fields"], [{"field":"b", "old":0, "new":1}])
        self.assertEqual(result["comparison"]["absolute_reduction"], 10.0)
        self.assertEqual(result["comparison"]["percentage_reduction"], 10.0)

    def test_ties_use_performance_then_original_position(self):
        rows = [{"a":"0", "b":"0", "Performance-":"10"},
                {"a":"1", "b":"0", "Performance-":"9"},
                {"a":"0", "b":"1", "Performance-":"9"}]
        self.assertEqual(summary.fix_rows(rows)["recommendation"]["position"], 2)

    def test_invalid_baseline_and_empty_and_no_improvement(self):
        self.assertEqual(summary.fix_rows([])["status"], "empty_input")
        self.assertEqual(summary.fix_rows([{"a":"2", "Performance-":"10"}])["status"], "invalid_baseline")
        self.assertEqual(summary.fix_rows([{"a":"0", "Performance-":"10"}, {"a":"1", "Performance-":"10"}])["status"], "no_improving_candidate")

    def test_invalid_rows_positions_and_zero_baseline(self):
        rows = [{"a":"0", "b":"0", "Performance-":"0"},
                {"a":"2", "b":"0", "Performance-":"-1"},
                {"a":"1", "b":"0", "Performance-":"-1"}]
        result = summary.fix_rows(rows)
        self.assertEqual(result["recommendation"]["position"], 3)
        self.assertIsNone(result["comparison"]["percentage_reduction"])
        self.assertEqual(result["excluded"][0]["position"], 2)

    def test_missing_and_nonfinite_outcomes(self):
        rows = [{"a":"0", "Performance-":"10"}, {"a":"1", "Performance-":"nan"}, {"a":"1", "Performance-":""}]
        result = summary.fix_rows(rows)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual([x["position"] for x in result["excluded"]], [2, 3])

if __name__ == "__main__":
    unittest.main()
