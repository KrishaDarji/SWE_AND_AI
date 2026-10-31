import copy, csv, os, tempfile, unittest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import estimator

class EstimatorTests(unittest.TestCase):
    def rows(self):
        xs = [("a",)*22, ("b",)*22, ("c",)*22, ("d",)*22]
        return [{"id": str(i+10), "position": i, "x": x, "y": float((i+1)*10)} for i,x in enumerate(xs)]
    def test_prediction_and_explanation_consistency(self):
        m = estimator.CategoricalNearestMedian(2).fit(self.rows()[:3]); e=m.explain(("a",)*22)
        self.assertEqual(m.predict(("a",)*22), e["estimate"]); self.assertTrue(all(n["label"]=="model neighbor" for n in e["evidence"]))
    def test_excluded_inputs_and_missing_unseen(self):
        m=estimator.CategoricalNearestMedian().fit(self.rows()[:2]); self.assertEqual(len(m.neighbors(("z",)*22)), 2)
        with self.assertRaises(ValueError): m.predict(("a",)*21)
    def test_training_only_reference_and_proposal_selection(self):
        rows=self.rows(); tr,te=estimator.grouped_split(rows,42); m=estimator.CategoricalNearestMedian().fit(tr)
        self.assertEqual(m.reference, estimator.median([r["y"] for r in tr])); self.assertEqual(min(te,key=lambda r:r["position"])["id"], "10")
    def test_outcome_change_does_not_change_pre_audit(self):
        rows=self.rows(); m=estimator.CategoricalNearestMedian().fit(rows[:3]); a=m.explain(rows[3]["x"]); changed=copy.deepcopy(rows); changed[3]["y"]=9999
        b=m.explain(changed[3]["x"]); self.assertEqual(a,b)
    def test_invalid_targets_and_original_identifier(self):
        d=tempfile.NamedTemporaryFile(mode="w",delete=False,newline=""); self.addCleanup(lambda: os.unlink(d.name))
        fields=["idX"]+list(estimator.DESCRIPTORS)+["Effort-"]; w=csv.DictWriter(d,fieldnames=fields); w.writeheader()
        for ident,val in [("bad","NaN"),("keep","12")]: w.writerow({"idX":ident,**{x:"h" for x in estimator.DESCRIPTORS},"Effort-":val})
        d.close(); rows=estimator.load_rows(d.name); self.assertEqual([r["id"] for r in rows],["keep"]); self.assertEqual(rows[0]["position"],1)
    def test_insufficient_data(self):
        with self.assertRaises(ValueError): estimator.grouped_split(self.rows()[:1])

if __name__ == "__main__": unittest.main()
