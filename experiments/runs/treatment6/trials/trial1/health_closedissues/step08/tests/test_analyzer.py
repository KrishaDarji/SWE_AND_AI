import csv, math, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyzer import summarize, CONFIG, OUTCOMES, DIRECTIONS, ConfigurationPredictor, welch_comparison, Flow, flow_rows, flow_csv, blame, justify, choose, recommend_fix

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
        b=ConfigurationPredictor(); b.fit(q,"MRE-"); self.assertEqual(p, a.preprocessing)
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

class FlowTests(unittest.TestCase):
    def rows(self):
        return [
            {"MRE-":"3", "ACC+":".2", "PRED40+":".5", "N_estimators":"1", "criterion":"a", "Min_sample_leaves":"", "Min_impurity_decrease":".1", "Max_depth":"2"},
            {"MRE-":"1", "ACC+":".2", "PRED40+":"NaN", "N_estimators":"2", "criterion":"b", "Min_sample_leaves":"2", "Min_impurity_decrease":".2", "Max_depth":"3"},
            {"MRE-":"NaN", "ACC+":".8", "PRED40+":".5", "N_estimators":"3", "criterion":"c", "Min_sample_leaves":"3", "Min_impurity_decrease":".3", "Max_depth":"4"},
        ]
    def test_prefixes_and_ties(self):
        rows = self.rows(); snapshots = list(flow_rows(iter(rows)))
        self.assertEqual([s["rows_seen"] for s in snapshots], [1, 2, 3])
        self.assertEqual(snapshots[2]["outcomes"]["MRE-"]["count"], 2)
        self.assertAlmostEqual(snapshots[2]["outcomes"]["MRE-"]["sample_variance"], 2.0)
        self.assertEqual(snapshots[2]["best"]["ACC+"]["position"], 3)
        self.assertEqual(snapshots[2]["best"]["PRED40+"]["position"], 1)
        self.assertIsNone(snapshots[0]["best"]["MRE-"]["configuration"]["Min_sample_leaves"])
        snapshots[2]["best"]["MRE-"]["configuration"]["criterion"] = "changed"
        self.assertEqual(snapshots[1]["best"]["MRE-"]["configuration"]["criterion"], "b")
    def test_empty_single_and_iterator_once(self):
        self.assertEqual(list(flow_rows(iter([]))), [])
        one = list(flow_rows(iter([self.rows()[0]])))[0]
        self.assertIsNone(one["outcomes"]["MRE-"]["sample_variance"])
        class Once:
            def __iter__(self):
                if hasattr(self, "used"): raise AssertionError("iterated twice")
                self.used = True; return iter(self.rows)
            rows = [{"MRE-":"1", "ACC+":"2", "PRED40+":"3"}]
        self.assertEqual(list(flow_rows(Once()))[0]["rows_seen"], 1)
    def test_csv_integration(self):
        path = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        path.write(HEADER + "\n1,.2,.3,1,a,2,.1,3\n"); path.close(); self.addCleanup(lambda: os.unlink(path.name))
        self.assertEqual(list(flow_csv(path.name))[-1]["rows_seen"], 1)

class BlameTests(unittest.TestCase):
    def base(self):
        return [{"MRE-": str(y), "ACC+": "9", "PRED40+": "8", "N_estimators": str(x),
                 "criterion": c, "Min_sample_leaves": str(x), "Min_impurity_decrease": str(x), "Max_depth": str(x)}
                for x, y, c in [(1, 1, "z"), (2, 1, "z"), (3, 3, "a"), (4, 3, "a")]]
    def test_known_eta_direction_and_outcome_exclusion(self):
        report = blame(self.base(), 2); fields = {x["field"]: x for x in report["fields"]}
        self.assertAlmostEqual(fields["N_estimators"]["eta_squared"], 1.0)
        self.assertEqual(fields["N_estimators"]["groups_low_to_high_mre"][0]["mean_mre"], 1)
        self.assertEqual([x["field"] for x in report["fields"]], sorted(CONFIG))
        self.assertNotIn("ACC+", report); self.assertNotIn("PRED40+", report)
    def test_missing_nonfinite_and_per_field_population(self):
        rows = self.base() + [{"MRE-": "NaN", "N_estimators": "5", "criterion": "z"},
                              {"MRE-": "4", "N_estimators": "NaN", "criterion": "z"}]
        fields = {x["field"]: x for x in blame(rows)["fields"]}
        self.assertEqual(fields["N_estimators"]["eligible_count"], 4)
        self.assertEqual(fields["criterion"]["eligible_count"], 5)
        self.assertEqual(fields["N_estimators"]["excluded_count"], 2)
        self.assertEqual(len(fields["N_estimators"]["boundaries"]), 3)
    def test_constant_empty_and_ties(self):
        constant = [{"MRE-": "2", **{k: "1" for k in CONFIG}}]
        self.assertEqual(blame(constant)["fields"][0]["status"], "constant_outcome_or_setting")
        self.assertEqual(blame([])["fields"][0]["status"], "empty")
        tied = [{"MRE-": "1", **{k: "1" for k in CONFIG}}, {"MRE-": "2", **{k: "2" for k in CONFIG}}]
        self.assertEqual([x["field"] for x in blame(tied)["fields"]], sorted(CONFIG))

class JustifyTests(unittest.TestCase):
    def row(self, pos, mre, acc, pred, **overrides):
        r = {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred),
             "N_estimators": str(pos), "criterion": "squared_error",
             "Min_sample_leaves": "2", "Min_impurity_decrease": ".1", "Max_depth": "5"}
        r.update({k: str(v) for k, v in overrides.items()}); return r
    def test_all_tiebreakers_and_full_retention(self):
        rows = [self.row(1, 2, .8, .9), self.row(2, 1, .5, .4),
                self.row(3, 1, .6, .4), self.row(4, 1, .6, .4)]
        r = justify(rows); self.assertEqual(r["recommendation"]["position"], 3)
        self.assertEqual(r["recommendation"]["configuration"]["Max_depth"], 5)
        self.assertEqual(r["selection_rule"], ["lowest MRE-", "highest ACC+", "highest PRED40+", "earliest original CSV row position"])
    def test_eligibility_positions_medians_and_directions(self):
        rows = [self.row(1, 1, 10, 2), self.row(2, 3, 20, 4), self.row(3, 5, 30, 6),
                self.row(4, "NaN", 99, 99), self.row(5, 0, 0, 0, criterion="")]
        r = justify(rows); self.assertEqual(r["recommendation"]["position"], 1)
        self.assertEqual(r["references"]["MRE-"]["count"], 4); self.assertEqual(r["references"]["MRE-"]["median"], 2)
        self.assertEqual(r["references"]["ACC+"]["count"], 5); self.assertEqual(r["direction_labels"]["MRE-"], "minimize")
        self.assertTrue(r["comparisons"]["MRE-"]["favorable_vs_median"])
    def test_empty_and_no_eligible(self):
        self.assertEqual(justify([])["status"], "empty")
        self.assertEqual(justify([self.row(1, 1, 2, 3, Max_depth="NaN")])["status"], "no_eligible_row")
        self.assertIsNone(justify([{"MRE-":"1"}])["recommendation"])

class ChooseTests(unittest.TestCase):
    def row(self, pos, mre=1, acc=.5, pred=.5, n=10, **overrides):
        r = {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": str(n),
             "criterion": "squared_error", "Min_sample_leaves": "2", "Min_impurity_decrease": ".1", "Max_depth": "5"}
        r.update({k: str(v) for k, v in overrides.items()}); return r
    def test_cap_counts_and_complete_retention(self):
        rows = [self.row(1, n=100), self.row(2, mre=0, n=101), self.row(3, n=-1),
                self.row(4, n="NaN"), self.row(5, Max_depth="NaN"), self.row(6, criterion=""), self.row(7, n=100)]
        r = choose(rows, 100)
        self.assertEqual(r["recommendation"]["position"], 1)
        self.assertEqual(r["raw_row_count"], 7); self.assertEqual(r["valid_row_count"], 3)
        self.assertEqual(r["feasible_row_count"], 2); self.assertEqual(r["invalid_exclusion_count"], 4)
        self.assertEqual(r["over_budget_exclusion_count"], 1)
        self.assertEqual(r["recommendation"]["configuration"]["Max_depth"], 5)
    def test_tie_breakers_independently(self):
        self.assertEqual(choose([self.row(1, 2), self.row(2, 1, .4), self.row(3, 1, .5)] )["recommendation"]["position"], 3)
        self.assertEqual(choose([self.row(1, 1, .5, .4), self.row(2, 1, .5, .6)])["recommendation"]["position"], 2)
        self.assertEqual(choose([self.row(1, 1, .5, .5), self.row(2, 1, .5, .5)])["recommendation"]["position"], 1)
    def test_invalid_caps_empty_and_no_feasible(self):
        for cap in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError): choose([], cap)
        self.assertEqual(choose([], 100)["status"], "empty")
        self.assertEqual(choose([self.row(1, n=101)], 100)["status"], "no_feasible_row")
    def test_over_budget_excellent_row_cannot_win(self):
        r = choose([self.row(1, mre=10, n=100), self.row(2, mre=0, n=101)], 100)
        self.assertEqual(r["recommendation"]["position"], 1)

class FixTests(unittest.TestCase):
    def row(self, pos, mre, acc=.5, pred=.5, **changes):
        r = {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": "10",
             "criterion": "squared_error", "Min_sample_leaves": "2", "Min_impurity_decrease": "0.1", "Max_depth": "5"}
        r.update({k: str(v) for k, v in changes.items()}); return r
    def test_minimum_changes_beats_magnitude_and_reports(self):
        r = recommend_fix([self.row(1, 10), self.row(2, 1, N_estimators=11),
                           self.row(3, 0, criterion="absolute_error", Max_depth=9)])
        self.assertEqual(r["recommendation"]["position"], 2)
        self.assertEqual(r["recommendation"]["changed_field_count"], 1)
        self.assertEqual(r["recommendation"]["changed_fields"][0]["old"], 10)
        self.assertEqual(r["recommendation"]["changed_fields"][0]["new"], 11)
    def test_tiebreakers_independent_and_numeric_normalization(self):
        base = self.row(1, 10)
        self.assertEqual(recommend_fix([base, self.row(2, 9, acc=.4, N_estimators="11.0"), self.row(3, 9, acc=.5, Max_depth=6)])["recommendation"]["position"], 3)
        self.assertEqual(recommend_fix([base, self.row(2, 9, acc=.5, pred=.4, N_estimators=11), self.row(3, 9, acc=.5, pred=.6, N_estimators=12)])["recommendation"]["position"], 3)
        self.assertEqual(recommend_fix([base, self.row(2, 9, acc=.5, pred=.5, N_estimators=11), self.row(3, 9, acc=.5, pred=.5, N_estimators=12)])["recommendation"]["position"], 2)
    def test_statuses_exclusions_secondary_worsening_and_no_cap(self):
        self.assertEqual(recommend_fix([])["status"], "empty")
        self.assertEqual(recommend_fix([self.row(1, "NaN"), self.row(2, 0)])["status"], "invalid_baseline")
        bad = self.row(2, 0); bad["ACC+"] = "inf"
        r = recommend_fix([self.row(1, 10, acc=.2, pred=.2), bad, self.row(3, 9, acc=.1, pred=.1, N_estimators=1000)])
        self.assertEqual(r["recommendation"]["position"], 3)
        self.assertLess(r["recommendation"]["secondary_outcome_differences"]["ACC+"], 0)
        self.assertEqual(recommend_fix([self.row(1, 1), self.row(2, 1)])["status"], "no_improving_candidate")
        self.assertEqual(recommend_fix([self.row(1, 1), self.row(2, 0, N_estimators=1000000)])["status"], "success")

if __name__ == "__main__": unittest.main()
