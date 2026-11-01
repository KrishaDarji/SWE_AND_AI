import csv, math, pathlib, tempfile, unittest
import sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa_summary as n

class TestSummary(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        f.write(text); f.close(); self.addCleanup(lambda: pathlib.Path(f.name).unlink(missing_ok=True)); return f.name
    def test_roles_frequencies_summaries(self):
        p = self.write("idX,prec,Kloc+,Effort-,Defects-\nA,h,1,2,3\nA,l,3,4,5\n")
        d=n.load(p); s=n.summarize(d)
        self.assertEqual(s["record_count"],2); self.assertEqual(s["roles"]["prec"],"categorical"); self.assertEqual(s["categorical_frequencies"]["prec"],{"h":1,"l":1}); self.assertEqual(s["duplicate_idX"],["A"]); self.assertEqual([x["position"] for x in d["projects"]],[2,3]); self.assertEqual(s["outcomes"]["Kloc+"]["median"],2); self.assertEqual(s["outcomes"]["Kloc+"]["sample_variance"],2)
    def test_missing_nonfinite_invalid_and_all_missing(self):
        p=self.write("Kloc+,Effort-,Defects-,prec\n?,bad,NaN,h\n,inf,?,l\n")
        s=n.summarize(n.load(p)); self.assertEqual(s["outcomes"]["Kloc+"]["count"],0); self.assertEqual(s["invalid_numeric"]["Effort-"],2); self.assertEqual(s["invalid_numeric"]["Defects-"],1); self.assertEqual(s["missing"]["Defects-"],1); self.assertEqual(s["categorical_frequencies"]["prec"],{"h":1,"l":1})
    def test_empty_header_malformed_blank(self):
        e=self.write(""); self.assertEqual(n.load(e)["record_count"] if "record_count" in n.load(e) else len(n.load(e)["projects"]),0)
        h=self.write("a,b\n"); self.assertEqual(len(n.load(h)["projects"]),0)
        m=self.write("a,b\n1\n\n2,3\n"); d=n.load(m); self.assertEqual(len(d["projects"]),1); self.assertEqual(d["blank_rows"],1); self.assertEqual(len(d["errors"]),1)
    def test_median_even_odd(self): self.assertEqual(n.median([3,1,2]),2); self.assertEqual(n.median([4,1,3,2]),2.5)

class TestGuess(unittest.TestCase):
    def row(self, i, value, token="h"):
        return {"values": {"idX": str(i), "Effort-": str(value), **{c: token for c in n.PREDICTORS}}}
    def test_schema_zero_and_mean(self):
        model=n.CategoricalKNN([self.row(1,0), self.row(2,10)])
        cfg={c:"h" for c in n.PREDICTORS}; self.assertEqual(model.predict(cfg),5)
        with self.assertRaises(ValueError): model.predict({c:"h" for c in n.PREDICTORS[:-1]})
        with self.assertRaises(ValueError): model.predict({**cfg,"extra":"x"})
    def test_group_split_and_reproducible(self):
        rows=[self.row(i,i,"h" if i<2 else "l") for i in range(4)]
        a,b=n.split_groups(rows); c,d=n.split_groups(rows)
        self.assertEqual([x["values"]["idX"] for x in a],[x["values"]["idX"] for x in c])
        self.assertTrue(set(tuple(x["values"][k] for k in n.PREDICTORS) for x in a).isdisjoint(set(tuple(x["values"][k] for k in n.PREDICTORS) for x in b)))
    def test_nonfinite_excluded_and_whitespace(self):
        r=self.row(1,0," h "); r["values"]["Effort-"]="NaN"
        self.assertEqual(n._cell("  "),"?"); self.assertIsNone(n._effort("inf"))
    def test_insufficient_groups(self):
        with self.assertRaises(ValueError): n.split_groups([self.row(1,1)])

class TestCertify(unittest.TestCase):
    def test_welch_direction_variance_and_threshold(self):
        x=n.welch_test([10,12,14],[7,8,9],1)
        self.assertEqual(x["difference"],-4); self.assertAlmostEqual(x["standard_error"],math.sqrt(4/3+1/3)); self.assertLess(x["df"],10); self.assertEqual(x["threshold"]["point_meets"],True)
    def test_t_reference_values_and_invalid_threshold(self):
        self.assertAlmostEqual(n.t_cdf(0,1),.5); self.assertAlmostEqual(n.t_cdf(1.9599639845,100),.9736,places=3)
        with self.assertRaises(ValueError): n.welch_test([1,2],[1,2],-1)
    def test_edge_cases(self):
        self.assertEqual(n.welch_test([1],[2])["status"],"insufficient samples")
        self.assertEqual(n.welch_test([1,1],[2,2])["status"],"zero standard error")

class TestFlow(unittest.TestCase):
    def row(self, position, effort, token="h"):
        return {"position": position, "values": {"idX": "dup", "Effort-": effort, **{c: token for c in n.PREDICTORS}}}
    def test_prefix_order_and_matched_errors(self):
        result = n.flow([self.row(2, "0"), self.row(3, "10", "l"), self.row(4, "20", "h")])
        self.assertIsNone(result["records"][0]["prediction"])
        self.assertEqual([r["history_count"] for r in result["records"]], [0, 1, 2])
        self.assertEqual(result["evaluated_count"], 2)
        self.assertEqual(len(result["model_errors"]), len(result["baseline_errors"]))
    def test_current_and_future_outcomes_do_not_change_prediction_or_snapshot(self):
        rows = [self.row(2, "1"), self.row(3, "2", "l"), self.row(4, "3", "h")]
        first = n.flow(rows); snapshot = first["records"][1].copy()
        rows[1]["values"]["Effort-"] = "999999"; rows[2]["values"]["Effort-"] = "bad"
        second = n.flow(rows)
        self.assertEqual(first["records"][1]["prediction"], second["records"][1]["prediction"])
        self.assertEqual(first["records"][1], snapshot)
    def test_invalid_zero_unseen_missing_duplicate_position(self):
        result = n.flow([self.row(10, "bad"), self.row(20, "0", "?"), self.row(30, "5", "x")])
        self.assertEqual(result["invalid_outcome_count"], 1)
        self.assertEqual(result["records"][1]["actual_revealed"], 0)
        self.assertEqual([r["position"] for r in result["records"]], [10, 20, 30])

class TestBlame(unittest.TestCase):
    def row(self, effort, prec="h", flex="a"):
        return {"values": {"Effort-": str(effort), "prec": prec, "flex": flex, **{c: "x" for c in n.PREDICTORS if c not in ("prec", "flex")}}}
    def test_scores_means_ss_and_directions(self):
        got = {x["descriptor"]: x for x in n.blame({"projects": [self.row(1, " h "), self.row(3, "h"), self.row(9, "l")]})}
        x = got["prec"]
        self.assertEqual(x["eligible_count"], 3); self.assertAlmostEqual(x["reference_mean"], 13/3)
        self.assertAlmostEqual(x["between_ss"], 98/3); self.assertAlmostEqual(x["total_ss"], 104/3)
        self.assertAlmostEqual(x["eta_squared"], 49/52); self.assertEqual(x["direction"]["higher_mean"], ["l"])
        self.assertEqual(x["direction"]["lower_mean"], ["h"])
    def test_missing_invalid_constant_and_ties(self):
        rows = [self.row(1, "h"), self.row("bad", "l"), self.row(3, " "), self.row(5, "h")]
        got = {x["descriptor"]: x for x in n.blame({"projects": rows})}
        self.assertEqual(got["prec"]["eligible_count"], 2); self.assertEqual(got["prec"]["excluded_count"], 2)
        self.assertEqual(got["flex"]["eta_squared"], 0.0)
        ranks = n.blame({"projects": [self.row(1), self.row(1)]})
        self.assertEqual([x["descriptor"] for x in ranks], sorted(n.PREDICTORS))
        self.assertTrue(all(x["eta_squared"] is None for x in ranks))

class TestJustify(unittest.TestCase):
    def row(self, position, effort, token):
        return {"position": position, "values": {"idX": "duplicate", "Effort-": str(effort), **{c: token for c in n.PREDICTORS}}}
    def test_training_only_tied_evidence_and_baseline(self):
        rows = [self.row(i, i, "a" if i < 3 else "b") for i in range(1, 6)]
        out = n.justify(rows)
        self.assertEqual(out["query"]["position"], min(x["position"] for x in out["test"]))
        self.assertTrue(all(x["position"] in [p["position"] for p in out["train"]] for x in out["neighbors"]))
        self.assertEqual(out["prediction"], sum(x["effort"] for x in out["neighbors"])/len(out["neighbors"]))
        self.assertEqual(out["training_mean"], sum(p["values"]["Effort-"] != "" and float(p["values"]["Effort-"]) for p in out["train"])/len(out["train"]))
    def test_audit_mutation_does_not_change_prediction_or_explanation(self):
        rows = [self.row(i, i, "a" if i < 3 else "b") for i in range(1, 6)]
        first = n.justify(rows); rows[0]["values"]["Effort-"] = "999999"
        second = n.justify(rows)
        self.assertEqual(first["query"]["position"], second["query"]["position"])
        self.assertEqual(first["prediction"], second["prediction"])
        self.assertEqual(first["neighbors"], second["neighbors"])
    def test_invalid_target_and_schema_are_explicit(self):
        rows = [self.row(i, "bad" if i == 1 else i, "a" if i < 3 else "b") for i in range(1, 6)]
        out = n.justify(rows); self.assertTrue(all(math.isfinite(x["effort"]) for x in out["neighbors"]))
        with self.assertRaises(ValueError): n.CategoricalKNN([]).predict({c: "a" for c in n.PREDICTORS})
    def test_exact_predictors(self):
        self.assertEqual(tuple(n.PREDICTORS), ("prec","flex","resl","team","pmat","rely","data","cplx","ruse","docu","time","stor","pvol","acap","pcap","pcon","apex","plex","ltex","tool","site","sced"))
        self.assertTrue(set(n.PREDICTORS).isdisjoint(n.METADATA | set(n.OUTCOMES) | {"MonthsX"}))

class TestChoose(unittest.TestCase):
    def row(self, pos, kloc, effort, defects=1, token="h", ident="dup"):
        return {"position": pos, "values": {"idX": ident, "Kloc+": str(kloc), "Effort-": str(effort), "Defects-": str(defects), **{c: token for c in n.PREDICTORS}}}
    def test_max_and_ties_and_complete_retention(self):
        rows = [self.row(20, 50, 10, 9), self.row(10, 60, 10, 8), self.row(5, 60, 10, 7)]
        out = n.choose({"projects": rows, "errors": []}, 10)
        self.assertEqual(out["winner"]["position"], 5); self.assertEqual(out["winner"]["values"]["idX"], "dup")
        self.assertEqual(set(out["winner"]["values"]) >= set(n.PREDICTORS), True)
    def test_cap_invalid_negative_nonfinite_and_counts(self):
        row = self.row(2, 9, 10)
        for cap in (-1, float("inf"), "10"):
            with self.assertRaises(ValueError): n.choose({"projects": [row]}, cap)
        out = n.choose({"projects": [row], "errors": [{"kind": "malformed_row"}]}, 10)
        self.assertEqual((out["raw_count"], out["valid_count"], out["feasible_count"], out["malformed_row_count"]), (1, 1, 1, 1))
    def test_invalid_outcomes_missing_descriptor_and_over_budget(self):
        rows = [self.row(2, 999, 101), self.row(3, 100, 100), self.row(4, 1, -1), self.row(5, 2, 1)]
        rows[3]["values"]["prec"] = "?"; rows[2]["values"]["Defects-"] = "NaN"
        out = n.choose({"projects": rows}, 100)
        self.assertEqual(out["winner"]["position"], 3); self.assertEqual(out["invalid_data_count"], 2); self.assertEqual(out["over_budget_count"], 1)
    def test_empty_and_no_feasible(self):
        self.assertIsNone(n.choose({"projects": []}, 0)["winner"])
        self.assertIsNone(n.choose({"projects": [self.row(2, 9, 1)]}, 0)["winner"])
if __name__ == "__main__": unittest.main()
