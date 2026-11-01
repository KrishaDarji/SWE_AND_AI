import csv, math, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyzer import summarize, CONFIG, OUTCOMES, DIRECTIONS, ConfigurationPredictor, welch_comparison

HEADER = ",".join(OUTCOMES + CONFIG)
class AnalyzerTests(unittest.TestCase):
    def run_case(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return summarize(f.name)
    def test_complete_reporting_path(self):
        r = self.run_case(HEADER + "\n1,0.5,1,10,squared_error,2,0.1,4\n 2 , NaN , ? ,10,squared_error,2,0.1,4\n3,0.7,0,20,absolute_error,3,0.2,5\n")
        self.assertEqual(r["record_count"], 3); self.assertEqual(r["column_details"]["MRE-"]["statistics"]["mean"], 2)
        self.assertEqual(r["column_details"]["ACC+"]["missing"], 1); self.assertEqual(r["column_details"]["criterion"]["frequencies"], {"absolute_error": 1, "squared_error": 2})
        self.assertEqual(r["configuration_counts"]["distinct"], 2); self.assertEqual(r["configuration_counts"]["duplicate_rows"], 1); self.assertEqual(r["configuration_counts"]["duplicate_rows_exact"], 0)
        self.assertEqual(r["optimization_directions"], DIRECTIONS); self.assertEqual(r["column_details"]["criterion"]["type"], "categorical")
    def test_empty_and_header_only(self):
        self.assertEqual(summarize(self.path(""))["record_count"], 0)
        self.assertTrue(summarize(self.path(HEADER + "\n"))["empty"])
    def path(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", delete=False); f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name

class PredictorTests(unittest.TestCase):
    def rows(self, n=20):
        return [{"MRE-": str(i/20), "ACC+": str(1-i/40), "PRED40+": "0.5",
                 "N_estimators": str(10+i%4), "criterion": "a" if i%2 else "b",
                 "Min_sample_leaves": str(1+i%3), "Min_impurity_decrease": str(i%5), "Max_depth": str(2+i%4)} for i in range(n)]
    def test_grouped_reproducible_and_prediction(self):
        a=ConfigurationPredictor(); b=ConfigurationPredictor(); ea=a.fit(self.rows(), "MRE-"); eb=b.fit(self.rows(), "MRE-"); q={k:self.rows()[0][k] for k in CONFIG}
        self.assertEqual(ea, eb); self.assertEqual(a.predict(q, "MRE-"), b.predict(q, "MRE-")); self.assertGreater(ea["test_count"], 0)
    def test_schema_missing_unseen_and_excluded_outcome(self):
        r=self.rows(); a=ConfigurationPredictor(); a.fit(r, "ACC+")
        q={k:r[0][k] for k in CONFIG}; q["criterion"]="unseen"; q["N_estimators"]="NaN"
        self.assertTrue(math.isfinite(a.predict(q, "ACC+")))
        with self.assertRaises(ValueError): a.predict({k:v for k,v in q.items() if k != "Max_depth"}, "ACC+")
        with self.assertRaises(ValueError): a.predict(dict(q, Extra="x"), "ACC+")
    def test_invalid_zero_and_insufficient(self):
        r=self.rows(); r[0]["MRE-"]="0"; self.assertEqual(ConfigurationPredictor().fit(r,"MRE-")["train_count"]+ConfigurationPredictor().fit(r,"MRE-")["test_count"],20)
        for x in ("PRED40+",):
            with self.assertRaises(ValueError): ConfigurationPredictor().fit(r,x)
        for row in r:
            for k in CONFIG: row[k] = "1"
        with self.assertRaises(ValueError): ConfigurationPredictor().fit(r,"MRE-")
    def test_preprocessing_is_training_only(self):
        r=self.rows(); a=ConfigurationPredictor(); a.fit(r,"MRE-"); p=a.preprocessing.copy(); q=list(r); q[-1]=dict(q[-1], N_estimators="999999", criterion="new")
        b=ConfigurationPredictor(); b.fit(q,"MRE-"); self.assertEqual(p, a.preprocessing); self.assertNotEqual(a.predict({k:r[0][k] for k in CONFIG},"MRE-"), b.predict({k:r[0][k] for k in CONFIG},"MRE-"))
class CertifyTests(unittest.TestCase):
    def rows(self, ref=(1,2,3,4), cand=(.5,1.5,2.5,3.5)):
        return [{"criterion":"squared_error","MRE-":str(x)} for x in ref] + [{"criterion":"absolute_error","MRE-":str(x)} for x in cand] + [{"criterion":"other","MRE-":"999"}]
    def test_fixed_selection_direction_and_ci(self):
        r=welch_comparison(self.rows()); self.assertEqual(r["groups"]["reference"]["count"],4); self.assertAlmostEqual(r["difference"],-.5); self.assertLess(r["confidence_interval_95"][0],r["difference"]); self.assertLess(r["difference"],r["confidence_interval_95"][1]); self.assertTrue(0<=r["p_value"]<=1); self.assertGreater(r["degrees_of_freedom"],1)
    def test_threshold_missing_and_insufficient(self):
        r=welch_comparison(self.rows(),.25); self.assertTrue(r["practical"]["point_estimate_exceeds"]); self.assertFalse(r["practical"]["entire_interval_supports"]); self.assertEqual(welch_comparison(self.rows((1,),(.5,)))["status"],"undefined")
        bad=self.rows()+[{"criterion":"absolute_error","MRE-":"NaN"}]; self.assertEqual(welch_comparison(bad)["groups"]["candidate"]["count"],4)
    def test_zero_variance_and_threshold_validation(self):
        r=welch_comparison(self.rows((1,1,1,1),(1,1,1,1))); self.assertEqual(r["status"],"undefined"); self.assertNotIn("p_value",r)
        with self.assertRaises(ValueError): welch_comparison(self.rows(),-1)

if __name__ == "__main__": unittest.main()
