import csv, math, os, tempfile, unittest, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from nasa93_summary import read_dataset, summarize, EffortGuesser, grouped_split, guess, certify, flow, blame, justify, choose, fix, spot, race, paired_difference, _t_quantile

class RaceTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def data(self, outcomes="10,20,30,40"):
        return "idX,a,b,Kloc+,Effort-,Defects-,noteX\n" + "\n".join(
            f"dup,{a},{b},99,{y},1,metadata" for (a,b),y in zip([(" x ","p"),("x","p"),("y","q"),("z","r")], outcomes.split(","))) + "\n"
    def test_paired_arithmetic_and_labels(self):
        self.assertEqual(paired_difference(2, 3)["classification"], "win")
        self.assertEqual(paired_difference(3, 2)["classification"], "loss")
        self.assertEqual(paired_difference(2, 2)["classification"], "tie")
        self.assertEqual(paired_difference(2, 3)["difference"], -1)
    def test_reproducible_group_separation_and_identifiers(self):
        p = self.fixture(self.data())
        a, b = race(p, seeds=[42, 43]), race(p, seeds=[42, 43])
        self.assertEqual(a, b); self.assertEqual(a["predictor_count"], 2)
        for split in a["splits"]:
            self.assertTrue(split["group_separation"])
            self.assertTrue(set(split["train_csv_positions"]).isdisjoint(split["test_csv_positions"]))
            self.assertTrue(all("idX" in item for item in split["projects"]))
    def test_training_mean_baseline_and_fresh_outcome_independence(self):
        original = self.data(); p = self.fixture(original)
        split = race(p, seeds=[42])["splits"][0]
        lines = original.splitlines(); held_out = split["test_csv_positions"][0]
        cells = lines[held_out].split(","); cells[4] = "999"; lines[held_out] = ",".join(cells)
        q = self.fixture("\n".join(lines) + "\n")
        a, b = race(p, seeds=[42]), race(q, seeds=[42])
        self.assertEqual(a["splits"][0]["training_mean"], b["splits"][0]["training_mean"])
        self.assertEqual([x["prediction"] for x in a["splits"][0]["projects"]], [x["prediction"] for x in b["splits"][0]["projects"]])
    def test_invalid_targets_excluded_and_insufficient_groups(self):
        p = self.fixture("idX,a,Effort-\n1,x,10\n2,y,?\n3,z,inf\n")
        with self.assertRaises(ValueError): race(p, seeds=[42])
        q = self.fixture("idX,a,Effort-\n1,x,10\n2,x,20\n")
        with self.assertRaises(ValueError): race(q, seeds=[42])

class SummaryTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_summaries_roles_and_directions(self):
        p = self.fixture("idX,kind,feature,Kloc+,Effort-,Defects-,bad\n1,a, 2 ,10,20,3,\n2,a,4,20,40,?,oops\n3,b,?,30,inf,5,7\n")
        s = summarize(p)
        self.assertEqual(s["record_count"], 3); self.assertEqual(s["categorical"]["kind"], {"a": 2, "b": 1})
        self.assertEqual(s["numeric"]["feature"]["count"], 2); self.assertEqual(s["missing"]["feature"], 1)
        self.assertEqual(s["invalid_numeric"]["bad"], 0); self.assertEqual(s["roles"]["idX"], "metadata")
        self.assertEqual(s["outcomes"]["Kloc+"]["direction"], "maximize")
        self.assertEqual(s["outcomes"]["Effort-"]["direction"], "minimize")
        self.assertNotIn("Kloc+", s["model_predictors"])
    def test_whitespace_invalid_and_header_only(self):
        p = self.fixture("A,b\n 1 , x \n nan, y\n2,\n")
        h, rows, _ = read_dataset(p); self.assertEqual(rows[0]["A"], "1"); self.assertEqual(summarize(p)["invalid_numeric"]["A"], 0)
        q = self.fixture("A,b\n"); self.assertEqual(summarize(q)["record_count"], 0)
        r = self.fixture(""); self.assertEqual(summarize(r)["columns"], [])
    def test_known_statistics(self):
        p = self.fixture("A,b\n1,x\n3,x\n5,y\n")
        self.assertEqual(summarize(p)["numeric"]["A"]["mean"], 3.0)
        self.assertEqual(summarize(p)["numeric"]["A"]["median"], 3.0)

class GuessTests(unittest.TestCase):
    def fixture(self, text):
        f=tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def rows(self):
        return [{"a": " x ", "b": "h", "Effort-": "10"}, {"a": "x", "b": "h", "Effort-": "20"},
                {"a": "y", "b": "l", "Effort-": "30"}, {"a": "z", "b": "l", "Effort-": "40"}]
    def test_grouped_reproducible_and_training_only(self):
        a,b=grouped_split(self.rows(), ["a","b"]); c,d=grouped_split(self.rows(), ["a","b"])
        self.assertEqual(a,c); self.assertEqual(b,d); self.assertTrue(set(map(id,a)).isdisjoint(map(id,b)))
        m=EffortGuesser(["a","b"]).fit(a); self.assertEqual(m.mean, statistics.fmean([float(r["Effort-"]) for r in a]))
    def test_missing_unseen_and_whitespace(self):
        m=EffortGuesser(["a","b"]).fit(self.rows()[:2]); self.assertEqual(m.predict([{ "a":"x","b":"h"}])[0],15)
        self.assertEqual(m.predict([{ "a":"new","b":"q"}])[0],15)
    def test_excluded_columns_and_proposal_validation(self):
        p=self.fixture("idX,a,Kloc+,Effort-,Defects-\n1,x,1,10,2\n2,y,2,20,3\n3,z,3,30,4\n4,w,4,40,5\n")
        out=guess(p); self.assertEqual(out["predictors"],["a"]); self.assertIn("baseline_mae",out)
        with self.assertRaises(ValueError): guess(p,{"wrong":"x"})
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): grouped_split([{ "a":"x"},{"a":"x"}], ["a"])

class CertifyTests(unittest.TestCase):
    def fixture(self, text):
        f=tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline=""); f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_welch_whitespace_direction_and_reference_t(self):
        p=self.fixture("tool,Effort-\n n ,10\nn,12\nn,14\nh,20\nh,22\nh,24\n")
        out=certify(p, 1); self.assertEqual(out["groups"]["A"]["count"],3); self.assertLess(out["mean_difference_A_minus_B"],0)
        self.assertAlmostEqual(_t_quantile(.975, 10), 2.22813885, places=5)
    def test_crossing_threshold_and_other_direction(self):
        p=self.fixture("tool,Effort-\nn,10\nn,40\nh,20\nh,20\n")
        out=certify(p, 4); self.assertTrue(out["point_estimate_exceeds_threshold"]); self.assertFalse(out["interval_entirely_practically_below_A"])
    def test_missing_zero_variance_and_invalid_threshold(self):
        p=self.fixture("tool,Effort-\nn,10\nn,?\nh,20\nh,inf\n")
        with self.assertRaises(ValueError): certify(p, 0)
        q=self.fixture("tool,Effort-\nn,10\nn,10\nh,10\nh,10\n")
        with self.assertRaises(ValueError): certify(q, 0)
        with self.assertRaises(ValueError): certify(q, -1)

class FlowTests(unittest.TestCase):
    def fixture(self, text):
        f=tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline=""); f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_warmup_baseline_errors_and_exclusions(self):
        p=self.fixture("idX,kind,Kloc+,Effort-,Defects-,noteX\n1, a ,999,10,1,old\n2,a,888,20,2,x\n3,a,777,30,3,y\n4,new,666,40,4,z\n")
        out=flow(p); self.assertEqual(out["predictors"],["kind"]); self.assertEqual(out["scored_count"],2)
        self.assertIsNone(out["records"][0]["prediction"]); self.assertIsNone(out["records"][1]["prediction"])
        self.assertEqual(out["records"][2]["prediction"],15.0); self.assertEqual(out["records"][2]["baseline"],15.0)
        self.assertEqual(out["records"][2]["absolute_error"],15.0); self.assertEqual(out["records"][3]["prediction"],20.0)
        self.assertEqual(out["records"][3]["cumulative_mae"],17.5)
    def test_invalid_effort_not_history_and_missing_unseen(self):
        p=self.fixture("idX,a,b,Effort-\n1,x,?,10\n2,x,y,?\n3,new,y,30\n4,z,z,inf\n5,new,y,50\n")
        out=flow(p); self.assertEqual(out["records"][2]["prior_training_count"],1)
        self.assertEqual(out["records"][4]["prior_training_count"],2); self.assertEqual(out["records"][4]["prediction"],30.0)
        self.assertEqual(out["scored_count"],1)
    def test_future_outcomes_do_not_change_earlier_prediction(self):
        a=self.fixture("idX,k,Effort-\n1,a,10\n2,a,20\n3,b,30\n")
        b=self.fixture("idX,k,Effort-\n1,a,10\n2,a,20\n3,b,777\n")
        x,y=flow(a),flow(b); self.assertEqual(x["records"][2]["prediction"],y["records"][2]["prediction"])
    def test_empty_and_short(self):
        self.assertEqual(flow(self.fixture("idX,k,Effort-\n"))["records"],[])
        out=flow(self.fixture("idX,k,Effort-\n1,a,10\n")); self.assertEqual(out["scored_count"],0)

class BlameTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_known_association_exclusions_and_determinism(self):
        p = self.fixture("idX,signal,noise,Kloc+,Effort-,Defects-,tagX\n1,low,a,1,10,2,x\n2,low,b,2,12,3,y\n3,high,a,3,90,4,z\n4,high,b,4,88,5,w\n")
        out = blame(p); self.assertEqual([x["descriptor"] for x in out["ranking"]], ["signal", "noise"])
        self.assertAlmostEqual(out["ranking"][0]["score"], .9993429697766097, places=12)
        self.assertEqual(out["ranking"][0]["groups"]["low"]["count"], 2)
        self.assertNotIn("Kloc+", out["predictors"]); self.assertNotIn("tagX", out["predictors"])
        self.assertEqual(out, blame(p))
    def test_categorical_constants_missing_small_groups_and_invalid_effort(self):
        p = self.fixture("a,b,Effort-\n x,one,10\n x,?,20\n x,two,?\n y,two,inf\n")
        out = blame(p); by = {x["descriptor"]: x for x in out["ranking"]}
        self.assertEqual(by["a"]["score"], 0.0); self.assertEqual(by["b"]["groups"]["<MISSING>"]["count"], 1)
        self.assertEqual(out["usable_effort_count"], 2)

class JustifyTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_frozen_explanation_excludes_metadata_and_audits_later(self):
        p = self.fixture("idX,kind, signalX, Kloc+,Effort-,Defects-\n"+
                        "bad,a,old,1,10,1\n"+
                        "r2,a,old,2,20,2\n"+
                        "r3,b,new,3,40,3\n"+
                        "r4,c,new,4,80,4\n")
        out = justify(p); frozen = out["pre_audit"]
        self.assertNotIn("Kloc+", frozen["predictors"]); self.assertNotIn("signalX", frozen["predictors"])
        self.assertEqual(frozen["record_identifier"], "r3")
        self.assertEqual(out["retrospective_audit"]["observed_effort"], 40.0)
        self.assertTrue(all("training rows only" in x["label"] for x in frozen["explanation"]["historical_associations_training_only"]))
    def test_future_outcome_does_not_change_frozen_report(self):
        a = self.fixture("idX,k,Effort-\n1,a,10\n2,a,20\n3,b,30\n4,c,40\n")
        b = self.fixture("idX,k,Effort-\n1,a,10\n2,a,20\n3,b,999\n4,c,40\n")
        self.assertEqual(justify(a)["pre_audit"], justify(b)["pre_audit"])
    def test_invalid_targets_before_selected_and_unseen_fallback(self):
        p = self.fixture("idX,k,Effort-\ninvalid,a,?\nvalid1,a,10\nvalid2,b,30\nvalid3,c,50\n")
        out = justify(p); self.assertIn(out["pre_audit"]["explanation"]["label"],
                                       ("exact training descriptor group", "training mean fallback for unseen group"))
        self.assertEqual(out["pre_audit"]["record_identifier"], "valid2")
    def test_insufficient_finite_targets(self):
        p = self.fixture("idX,k,Effort-\n1,a,?\n2,b,inf\n")
        with self.assertRaises(ValueError): justify(p)

class ChooseTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_boundaries_maximum_and_metadata_separation(self):
        p = self.fixture("idX,mode,effortX,Kloc+,Effort-,Defects-\na,red,meta,10,5,1\nb,blue,meta,30,10,2\nc,green,meta,20,10,3\nd,late,meta,99,11,4\n")
        out = choose(p, 10)
        self.assertEqual(out["record_identifier"], "b"); self.assertEqual(out["csv_position"], 2)
        self.assertEqual(out["feasible_count"], 3); self.assertEqual(out["total_count"], 4)
        self.assertEqual(out["descriptor_settings"], {"mode": "blue"})
        self.assertNotIn("idX", out["descriptor_settings"]); self.assertNotIn("Kloc+", out["descriptor_settings"])
        self.assertEqual(out["delivered_size_proxy"], 30.0)
    def test_tie_break_lower_effort_then_earliest_and_duplicate_ids(self):
        p = self.fixture("idX,setting,Kloc+,Effort-\ndup,a,50,9\ndup,b,50,8\ndup,c,50,8\n")
        out = choose(p, 9); self.assertEqual(out["record_identifier"], "dup"); self.assertEqual(out["csv_position"], 2)
    def test_invalid_missing_nonfinite_empty_and_infeasible(self):
        p = self.fixture("idX,s,Kloc+,Effort-\nbad,a,?,1\ninf,b,20,inf\nlate,c,30,5\n")
        with self.assertRaises(ValueError): choose(p, -1)
        with self.assertRaises(ValueError): choose(p, float("inf"))
        with self.assertRaises(ValueError): choose(p, 4)
        with self.assertRaises(ValueError): choose(self.fixture("idX,s,Kloc+,Effort-\n"), 10)
        self.assertEqual(choose(p, 5)["record_identifier"], "late")

class FixTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_minimum_change_and_both_tiebreakers_training_only(self):
        p = self.fixture("idX,a,b,Effort-\norig,x,y,90\nfirst,x,z,10\nlow,w,y,20\nsecond,w,z,10\nhold,q,q,999\n")
        out = fix(p, {"a":" x ", "b":" y "}, 25)
        self.assertEqual(out["status"], "repair-suggested")
        self.assertEqual(out["candidate_descriptors"], {"a":"x", "b":"z"})
        self.assertEqual(out["change_count"], 1); self.assertEqual(out["candidate_estimated_effort"], 10.0)
        self.assertEqual(out["candidate_record_identifier"], "first")
        self.assertEqual(out["changed_fields"], [{"field":"b", "old":"y", "new":"z"}])
        self.assertEqual(out["candidate_csv_position"], 2)
    def test_already_feasible_and_no_feasible(self):
        p = self.fixture("idX,a,Effort-\norig,x,10\nother,y,20\n")
        self.assertEqual(fix(p, {"a":"x"}, 10)["status"], "no-repair-needed")
        self.assertEqual(fix(p, {"a":"new"}, 5)["status"], "no-feasible-repair")
    def test_invalid_inputs_and_exact_descriptors(self):
        p = self.fixture("idX,a,Effort-\n1,x,10\n2,y,20\n")
        for budget in (-1, float("inf")):
            with self.assertRaises(ValueError): fix(p, {"a":"x"}, budget)
        with self.assertRaises(ValueError): fix(p, {}, 10)
        with self.assertRaises(ValueError): fix(p, {"a":"x", "Effort-":"10"}, 10)
        with self.assertRaises(ValueError): fix(self.fixture("idX,a,Effort-\n1,x,?\n"), {"a":"x"}, 10)
    def test_held_out_outcome_does_not_change_repair_and_metadata_identifier(self):
        a = self.fixture("idX,a,Effort-\nheld,x,500\ntrain1,y,10\ntrain2,z,20\n")
        b = self.fixture("idX,a,Effort-\nheld,x,500\ntrain1,y,999\ntrain2,z,20\n")
        proposal = {"a":"x"}
        self.assertEqual(fix(a, proposal, 25), fix(b, proposal, 25))
        self.assertEqual(fix(a, proposal, 25)["candidate_record_identifier"], "train2")

class SpotTests(unittest.TestCase):
    def fixture(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: os.unlink(f.name)); return f.name
    def test_unusual_values_counts_and_duplicate_identifiers(self):
        p = self.fixture("idX,Kloc+,Effort-,Defects-\ndup,10,10,1\ndup,11,11,1\ndup,12,12,1\nout,10000,10000,10000\n")
        out = spot(p, 3.5)
        self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertEqual(out["per_outcome_flag_counts"], {"Kloc+": 1, "Effort-": 1, "Defects-": 1})
        self.assertEqual(out["flags"][0]["identifier"], "out")
        self.assertEqual(out["flags"][0]["csv_position"], 4)
        self.assertTrue(all(x["score"] is None or x["score"] >= 3.5 for x in out["flags"][0]["flags"]))
    def test_invalid_negative_missing_nonfinite_and_strict_json(self):
        p = self.fixture("idX,Kloc+,Effort-,Defects-\na,-1,?,inf\nb,10,20,2\n")
        out = spot(p)
        text = __import__("json").dumps(out, allow_nan=False)
        self.assertIn("negative value", str(out)); self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertIn('"flags": [', text)
    def test_small_empty_constant_and_zero_scale(self):
        empty = self.fixture("idX,Kloc+,Effort-,Defects-\n")
        self.assertEqual(spot(empty)["unique_flagged_record_count"], 0)
        one = self.fixture("idX,Kloc+,Effort-,Defects-\na,1,1,1\n")
        self.assertEqual(spot(one)["per_outcome_flag_counts"], {"Kloc+": 0, "Effort-": 0, "Defects-": 0})
        constant = self.fixture("idX,Kloc+,Effort-,Defects-\na,1,1,1\nb,1,1,1\nc,2,2,2\n")
        out = spot(constant)
        self.assertEqual(out["unique_flagged_record_count"], 1)
        self.assertTrue(all(x["score"] is None for x in out["flags"][0]["flags"]))
    def test_threshold_validation(self):
        with self.assertRaises(ValueError): spot(self.fixture("idX,Kloc+\na,1\n"), -1)

if __name__ == "__main__": unittest.main()
