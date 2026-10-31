import json, math, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from spot_anomaly import detect

class SpotTests(unittest.TestCase):
    def test_normal_and_anomalous(self):
        rows = [{"x":"a","y":"0","Performance-":str(v)} for v in (10,11,9)]
        rows += [{"x":"b","y":"0","Performance-":str(v)} for v in (10,11,9)]
        rows += [{"x":"c","y":"0","Performance-":"100"}]
        out = detect(rows, min_groups=2)
        self.assertEqual(out["flagged_count"], 1)
        self.assertTrue(out["records"][-1]["flagged"])
    def test_isolation_and_duplicate_group_exclusion(self):
        rows = [{"x":"a","Performance-":"1"},{"x":"a","Performance-":"100"}]
        out = detect(rows, min_groups=1)
        self.assertEqual(out["records"][0]["reason"], "no independent configuration reference")
    def test_empty_small_constant_zero_variability(self):
        self.assertEqual(detect([])["reason"], "empty input")
        out = detect([{"x":"a","Performance-":"1"},{"x":"b","Performance-":"1"}], min_groups=1)
        self.assertEqual(out["records"][0]["reason"], "zero comparison variability")
    def test_missing_nonfinite_and_strict_json(self):
        rows = [{"x":"a","Performance-":v} for v in ("?", "nan", "inf", "-inf", "")]
        rows += [{"x":"b","Performance-":"2"}]
        out = detect(rows, min_groups=1)
        self.assertIsNone(out["records"][0]["score"])
        self.assertTrue(all(r["observed"] is None for r in out["records"][:5]))
        json.dumps(out, allow_nan=False)
    def test_actual_usable_group_count_and_identifiers(self):
        rows = [{"x":x,"Performance-":"1"} for x in ("a","b","c","d")]
        out = detect(rows, min_groups=1)
        self.assertEqual(out["records"][0]["comparison_group_count"], 3)
        self.assertEqual({g["x"] for g in out["records"][0]["usable_reference_groups"]}, {"b", "c", "d"})
    def test_invalid_nearby_groups_do_not_satisfy_minimum(self):
        rows = [{"x":"target","Performance-":"10"},
                {"x":"near1","Performance-":"11"},
                {"x":"near2","Performance-":"nan"},
                {"x":"near3","Performance-":"?"},
                {"x":"far","Performance-":"12"}]
        out = detect(rows, min_groups=3)
        rec = out["records"][0]
        self.assertEqual(rec["selected_group_count"], 4)
        self.assertEqual(rec["comparison_group_count"], 2)
        self.assertEqual([g["x"] for g in rec["usable_reference_groups"]], ["near1", "far"])
        self.assertEqual(rec["reason"], "insufficient independent comparison groups")

if __name__ == "__main__": unittest.main()
