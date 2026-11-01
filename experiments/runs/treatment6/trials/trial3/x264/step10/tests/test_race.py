import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class RaceTests(unittest.TestCase):
    def rows(self):
        return [{"a":"0", "b":"0", "Performance-":"bad"},
                {"a":"0.0", "b":"0", "Performance-":"9"},
                {"a":"1", "b":"0", "Performance-":"2"},
                {"a":"0", "b":"1", "Performance-":"3"},
                {"a":"1", "b":"1", "Performance-":"?"}]

    def test_same_pool_budget_charging_and_equivalence(self):
        r = summary.race_rows(self.rows(), budget=20, seeds=[42, 43], audit=False)
        self.assertEqual(r["pool_size"], 4)
        self.assertTrue(r["strategy_equivalence"])
        for pair in r["pairs"]:
            self.assertEqual(pair["selector"]["attempted_evaluations"], 4)
            self.assertEqual(pair["baseline"]["attempted_evaluations"], 4)
            self.assertEqual(pair["selector"]["successful_evaluations"], 2)
            self.assertEqual(pair["baseline"]["successful_evaluations"], 2)
            self.assertEqual(pair["selector"]["evaluation_history"], pair["baseline"]["evaluation_history"])

    def test_classification_boundaries_and_summary(self):
        self.assertEqual(summary._race_classification(1.0, 1.0), "tie")
        self.assertEqual(summary._race_classification(-1.01, 1.0), "loss")
        self.assertEqual(summary._race_classification(1.01, 1.0), "win")
        pairs = [{"difference": 2.0, "classification":"win"},
                 {"difference": 0.0, "classification":"tie"},
                 {"difference": -2.0, "classification":"loss"},
                 {"difference": None, "classification":"uncomparable"}]
        s = summary._race_summary(pairs, 0)
        self.assertEqual((s["mean_difference"], s["median_difference"], s["comparable_pairs"]), (0.0, 0.0, 3))
        self.assertEqual((s["wins"], s["ties"], s["losses"], s["uncomparable_pairs"]), (1,1,1,1))

    def test_edges_reproducibility_and_audit(self):
        self.assertEqual(summary.race_rows(self.rows(), 0, [42], audit=False)["pairs"][0]["selector"]["status"], "zero_budget")
        self.assertEqual(summary.race_rows([], 20, [42], audit=False)["pool_size"], 0)
        a = summary.race_rows(self.rows(), 2, [42], audit=False)
        b = summary.race_rows(self.rows(), 2, [42], audit=True)
        self.assertEqual(a["pairs"][0]["selector"], b["pairs"][0]["selector"])
        self.assertIn("retrospective_comparisons", b)
        with self.assertRaises(ValueError): summary.race_rows(self.rows(), 1, [42], -1)
        with self.assertRaises(ValueError): summary.race_rows(self.rows(), 1, [42], float("nan"))

if __name__ == "__main__": unittest.main()
