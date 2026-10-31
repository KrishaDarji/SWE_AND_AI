import json, math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from spot import detect

F = ["a", "b", "c"]
def row(a, b, c, y): return {"a": a, "b": b, "c": c, "Performance-": y}

class SpotTests(unittest.TestCase):
    def test_normal_and_anomaly(self):
        rows = [row(0,0,0,10), row(0,0,1,10), row(0,1,0,10), row(1,0,0,10),
                row(1,1,1,100)]
        out = detect(rows, F, min_groups=3)
        self.assertEqual(out["flagged_record_ids"], [5])
        self.assertEqual(out["unique_flagged_record_count"], 1)

    def test_outcome_isolation_and_duplicate_exclusion(self):
        rows = [row(0,0,0,0), row(0,0,0,1000), row(0,0,1,10),
                row(0,1,0,10), row(1,0,0,10)]
        out = detect(rows, F, min_groups=3)
        self.assertEqual(out["records"][0]["expected_reference"], 10)
        self.assertEqual(out["records"][0]["comparison_group_count"], 3)
        self.assertEqual(out["records"][0]["reason"], "zero_reference_variability_divergent_observation_score_unbounded")
        self.assertEqual(out["records"][0]["score"], None)

    def test_small_empty_constant_zero_variability(self):
        self.assertEqual(detect([], F)["unique_flagged_record_count"], 0)
        out = detect([row(0,0,0,1)], F)
        self.assertEqual(out["records"][0]["reason"], "no_independent_reference_groups")
        rows = [row(0,0,0,1), row(0,0,1,1), row(0,1,0,1), row(1,0,0,1)]
        self.assertEqual(detect(rows, F)["records"][0]["reason"], "zero_or_unavailable_reference_variability")

    def test_strict_json_and_actual_count(self):
        rows = [row(0,0,0,1), row(0,0,1,2), row(0,1,0,3), row(1,0,0,4), row(1,1,1,5)]
        out = detect(rows, F)
        json.dumps(out, allow_nan=False)
        self.assertEqual(out["records"][4]["comparison_group_count"], 3)
        self.assertTrue(all(math.isfinite(x["score"]) for x in out["records"] if x["score"] is not None))

if __name__ == "__main__": unittest.main()
