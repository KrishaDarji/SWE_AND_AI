import csv, os, tempfile, unittest, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_summary import summarize, split_groups, NearestNeighborModel, load, certify, _t_cdf, FlowAccumulator, flow, iter_measurements, blame

class SummaryTests(unittest.TestCase):
    def run_case(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            handle.write(text); handle.close(); return summarize(handle.name)
        finally:
            os.unlink(handle.name)
    def test_real_shape_and_stats(self):
        result = summarize(os.environ["X264_DATASET"])
        self.assertGreater(result["record_count"], 0); self.assertEqual(result["outcome"], "Performance-")
        self.assertEqual(result["direction"], "lower is better"); self.assertEqual(result["invalid_outcomes"], 0)
    def test_frequencies_statistics(self):
        result = self.run_case("a,b,Performance-\n1,x,2\n1,y,4\n0,x,6\n")
        self.assertEqual(result["frequencies"]["a"], {"1": 2, "0": 1})
        self.assertEqual(result["outcome_stats"]["count"], 3); self.assertEqual(result["outcome_stats"]["median"], 4)
    def test_missing_invalid(self):
        result = self.run_case("a,Performance-\n?,\n1,bad\n")
        self.assertEqual(result["missing_outcomes"], 1); self.assertEqual(result["invalid_outcomes"], 1)
    def test_empty_header_only_and_no_outcome(self):
        self.assertTrue(summarize(tempfile.NamedTemporaryFile(mode="w", delete=False).name)["missing_input"])
        self.assertEqual(self.run_case("a,b\n")["record_count"], 0)
        self.assertIsNone(self.run_case("a\n1\n")["outcome"])

if __name__ == "__main__": unittest.main()

class PredictionTests(unittest.TestCase):
    def test_prediction_grouping_validation_and_zero(self):
        text = "a,b,Performance-\n0,x,0\n0,x,2\n1,x,10\n1,y,12\n2,y,20\n"
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            handle.write(text); handle.close(); _, rows, _ = load(handle.name)
            train, test = split_groups(rows, ["a", "b"]); self.assertTrue(train and test)
            model = NearestNeighborModel(["a", "b"], "Performance-").fit(train)
            self.assertIsInstance(model.predict_one({"a": "0", "b": "x"}), float)
            with self.assertRaises(ValueError): model.predict_one({"a": "0"})
            self.assertEqual(model.predict_one({"a": "9", "b": "x"}), model.mean)
        finally: os.unlink(handle.name)

    def test_outcome_cannot_be_a_feature_and_zero_is_valid(self):
        model = NearestNeighborModel(["a"], "Performance-").fit([{"a":"x", "Performance-":"0"}, {"a":"y", "Performance-":"2"}])
        self.assertEqual(model.predict_one({"a":"x"}), 0)
        with self.assertRaises(ValueError): model.predict_one({"a":"x", "Performance-":"9", "extra":"z"})

class CertifyTests(unittest.TestCase):
    def case(self, text, threshold=None):
        h=tempfile.NamedTemporaryFile(mode="w", delete=False); h.write(text); h.close()
        try: return certify(h.name, threshold)
        finally: os.unlink(h.name)
    def test_direction_variance_and_inference(self):
        r=self.case("no_8x8dct,Performance-\n0,10\n0,14\n0,18\n1,4\n1,8\n1,12\n")
        self.assertEqual(r["reference"]["variance"],16); self.assertEqual(r["candidate"]["mean"],8); self.assertLess(r["inference"]["difference"],0)
        self.assertAlmostEqual(r["inference"]["standard_error"],math.sqrt(16/3+16/3))
    def test_threshold_and_exclusions(self):
        r=self.case("no_8x8dct,Performance-\n0,9\n0,11\n1,3\n1,5\n2,99\n1,bad\n",5)
        self.assertTrue(r["practical"]["point_estimate_meets"]); self.assertEqual(r["excluded"],2)
    def test_undefined_cases_and_probability(self):
        r=self.case("no_8x8dct,Performance-\n0,1\n1,2\n"); self.assertIsNone(r["inference"])
        r=self.case("no_8x8dct,Performance-\n0,1\n0,1\n1,1\n1,1\n"); self.assertIsNone(r["inference"])
        self.assertAlmostEqual(_t_cdf(0,1),.5); self.assertGreater(_t_cdf(2,10),.9); self.assertLess(_t_cdf(-2,10),.1)

class FlowTests(unittest.TestCase):
    def case(self, text):
        h=tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False); h.write(text); h.close()
        self.addCleanup(lambda: os.unlink(h.name)); return h.name
    def test_prefixes_and_edge_cases(self):
        path=self.case("a,b,Performance-\nfirst,x,0\nsecond,y,4\nthird,z,4\nfourth,q,bad\nfifth,w,?\n")
        acc=None; values=[]
        for pos, header, row in iter_measurements(path):
            acc = acc or FlowAccumulator(header); snap=acc.update(pos,row)
            if row["Performance-"] in ("0", "4"): values.append(float(row["Performance-"]))
            mean=sum(values)/len(values) if values else None
            variance=sum((x-mean)**2 for x in values)/(len(values)-1) if len(values)>1 else None
            self.assertEqual((snap["rows_seen"],snap["valid_count"],snap["mean"],snap["sample_variance"]), (pos,len(values),mean,variance))
        self.assertEqual(snap["best"]["position"], 1); self.assertEqual(snap["best"]["configuration"], {"a":"first","b":"x"})
        empty=FlowAccumulator().snapshot(); self.assertIsNone(empty["mean"]); self.assertIsNone(empty["sample_variance"]); self.assertIsNone(empty["best"])
    def test_snapshot_independence(self):
        acc=FlowAccumulator(["a", "Performance-"])
        first=acc.update(1,{"a":"x", "Performance-":"2"}); acc.update(2,{"a":"y", "Performance-":"1"})
        self.assertEqual(first["rows_seen"],1); self.assertEqual(first["best"]["value"],2.0)
    def test_csv_integration(self):
        result=flow(os.environ["X264_DATASET"]); self.assertGreater(result["rows_seen"],0); self.assertEqual(result["valid_count"],result["rows_seen"])

class BlameTests(unittest.TestCase):
    def case(self, text):
        h=tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False); h.write(text); h.close()
        self.addCleanup(lambda: os.unlink(h.name)); return blame(h.name)
    def test_eta_directions_counts_and_population(self):
        r=self.case("a,b,Performance-\n0,x,1\n0,y,3\n1,x,5\n1,y,7\n")
        a=next(x for x in r["settings"] if x["setting"] == "a")
        self.assertAlmostEqual(a["eta_squared"], .8); self.assertEqual(a["eligible"],4)
        self.assertEqual(a["higher_mean_groups"],["1"]); self.assertEqual(a["lower_mean_groups"],["0"])
        self.assertEqual([(g["value"],g["count"]) for g in a["groups"]],[('0',2),('1',2)])
    def test_normalization_missing_nonfinite_and_per_setting_population(self):
        r=self.case("a,b,Performance-\n0.0,x,1\n1,y,3\n?,z,5\n0,bad,NaN\n0,x,7\n")
        a=next(x for x in r["settings"] if x["setting"] == "a"); b=next(x for x in r["settings"] if x["setting"] == "b")
        self.assertEqual(a["eligible"],3); self.assertEqual(a["excluded"],2)
        self.assertEqual(b["eligible"],4); self.assertEqual(b["excluded"],1)
        self.assertEqual([g["value"] for g in a["groups"]],["0","1"])
    def test_constant_empty_insufficient_ties_and_outcome_exclusion(self):
        r=self.case("a,b,Performance-\n0,x,1\n0,x,1\n")
        self.assertTrue(all(x["eta_squared"] is None for x in r["settings"]))
        empty=self.case("a,Performance-\n"); self.assertIsNone(empty["settings"][0]["reference_mean"])
        tied=self.case("z,a,Performance-\n0,1,1\n1,1,2\n")
        self.assertEqual([x["setting"] for x in tied["settings"]], ["z","a"])
        self.assertNotIn("Performance-", [x["setting"] for x in tied["settings"]])
