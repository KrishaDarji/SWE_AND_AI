import json, math, os, sys, unittest
HERE = os.path.dirname(__file__); sys.path.insert(0, os.path.join(HERE, "..", "source"))
from anomaly_detector import detect

def row(x, y): return {"preset": x, "ref": x, "Performance-": y}

class DetectorTests(unittest.TestCase):
    def test_normal_and_anomalous(self):
        rows = [row("a", 10), row("b", 10), row("c", 11), row("d", 9), row("e", 100)]
        result = detect(rows, min_comparisons=4)
        self.assertEqual(result["flagged_record_count"], 1); self.assertEqual(result["flags"][0]["record_id"], 5)
        self.assertAlmostEqual(result["flags"][0]["expected_reference"], 10.0)
        self.assertAlmostEqual(result["flags"][0]["score"], 90 / math.sqrt(2 / 3), places=8)
    def test_group_isolation_and_reference_ids(self):
        rows = [row("a", 10), row("a", 999), row("b", 10), row("c", 11), row("d", 9)]
        first = detect(rows, min_comparisons=3)
        changed = list(rows); changed[1] = row("a", -999)
        second = detect(changed, min_comparisons=3)
        self.assertEqual(first["flags"][0]["expected_reference"], second["flags"][0]["expected_reference"])
        self.assertEqual(first["flags"][0]["reference_group_ids"], second["flags"][0]["reference_group_ids"])
        self.assertEqual(first["flags"][0]["comparison_count"], second["flags"][0]["comparison_count"])
    def test_distinct_group_minimum(self):
        self.assertEqual(detect([row("a", 1), row("b", 2)], min_comparisons=2)["skipped_record_count"], 2)
    def test_small_empty_constants_and_zero_variability(self):
        self.assertEqual(detect([])["status"], "empty_input")
        self.assertEqual(detect([row("a", 1), row("a", 1), row("b", 1)])["flagged_record_count"], 0)
        self.assertEqual(detect([row("a", 1)])["skipped_record_count"], 1)
    def test_strict_json_and_counts(self):
        result = detect([row("a", 1), row("b", 2), row("c", 50)], min_comparisons=2)
        encoded = json.dumps(result, allow_nan=False); self.assertNotIn("NaN", encoded); self.assertEqual(result["flags"][0]["comparison_count"], 2)
        self.assertTrue(all(math.isfinite(x["score"]) for x in result["flags"]))

    def test_missing_nonfinite_and_zero_variability(self):
        result = detect([row("a", 1), row("b", None), row("c", float("nan")), row("d", 1)], min_comparisons=2)
        self.assertEqual(result["skipped_record_count"], 4)
        result = detect([row("a", 1), row("b", 1), row("c", 1)], min_comparisons=2)
        self.assertEqual(result["flags"], [])

if __name__ == "__main__": unittest.main()
