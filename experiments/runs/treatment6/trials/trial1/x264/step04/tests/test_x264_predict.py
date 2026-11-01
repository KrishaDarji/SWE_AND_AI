import csv, os, tempfile, unittest
from source.x264_predict import _read, _split, Predictor, fit_evaluate, remember

class PredictTests(unittest.TestCase):
    def fixture(self, rows):
        h = tempfile.NamedTemporaryFile("w", newline="", delete=False, suffix=".csv")
        w = csv.writer(h); w.writerow(["a", "b", "Performance-"]); w.writerows(rows); h.close(); self.addCleanup(lambda: os.unlink(h.name)); return h.name
    def rows(self): return [[0,0,10],[0,0,11],[0,1,20],[0,1,21],[1,0,30],[1,0,31],[1,1,40],[1,1,41]]
    def test_grouped_reproducible_and_evaluated(self):
        path=self.fixture(self.rows()); names, rows=_read(path); a,b=_split(rows); c,d=_split(rows)
        self.assertEqual(a,c); self.assertEqual(b,d); self.assertTrue({x[0] for x in a}.isdisjoint({x[0] for x in b}))
        model, report=fit_evaluate(path); self.assertGreater(report["train_groups"],0); self.assertGreater(report["test_groups"],0); self.assertEqual(report["train_rows"]+report["test_rows"],8)
        self.assertEqual(model.predict({"a":"0","b":"0"}), model.predict({"a":0,"b":0}))
    def test_schema_and_bad_inputs(self):
        model, _=fit_evaluate(self.fixture(self.rows()))
        for x in ({"a":0}, {"a":0,"b":0,"c":1}, {"a":"bad","b":0}, {"a":float("nan"),"b":0}):
            with self.assertRaises(ValueError): model.predict(x)
    def test_training_only_fit_and_outcome_change(self):
        names, rows=_read(self.fixture(self.rows())); train, test=_split(rows)
        first=Predictor(names, train)
        changed=[(s, y+999999) for s,y in test]
        second=Predictor(names, train)
        self.assertEqual(first.predict({"a":0,"b":0}), second.predict({"a":0,"b":0}))
    def test_insufficient_groups_and_memory(self):
        with self.assertRaises(ValueError): fit_evaluate(self.fixture([[0,0,1],[0,0,2]]))
        box={}; self.assertEqual(remember(box,"x",3),3); self.assertEqual(box["x"],3)

if __name__ == "__main__": unittest.main()
