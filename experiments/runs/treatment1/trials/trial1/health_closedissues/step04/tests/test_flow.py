import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from flow import ConfigurationFlow, read_csv
class FlowTests(unittest.TestCase):
    def row(self, mre, acc, pred, n=10):
        return {"MRE-": str(mre), "ACC+": str(acc), "PRED40+": str(pred), "N_estimators": str(n), "criterion": "squared_error", "Min_sample_leaves": "1", "Min_impurity_decrease": "0.0", "Max_depth": "5"}
    def test_updates_best_and_average(self):
        f=ConfigurationFlow(); f.add(self.row(.4,.7,.2)); f.add(self.row(.2,.6,.8,20)); r=f.snapshot()
        self.assertEqual(r["records_seen"],2); self.assertAlmostEqual(r["average_metrics"]["MRE-"],.3); self.assertEqual(r["best_configurations"]["MRE-"]["N_estimators"],"20"); self.assertEqual(r["best_configurations"]["ACC+"]["N_estimators"],"10")
    def test_reader(self):
        with tempfile.NamedTemporaryFile("w",newline="",delete=False) as h:
            h.write("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n0.1,0.8,0.5,10,squared_error,1,0,5\n"); p=h.name
        try: self.assertEqual(read_csv(p)["records_seen"],1)
        finally: os.unlink(p)
    def test_missing_field(self):
        with self.assertRaises(ValueError): ConfigurationFlow().add({"MRE-":".1"})
if __name__ == "__main__": unittest.main()
