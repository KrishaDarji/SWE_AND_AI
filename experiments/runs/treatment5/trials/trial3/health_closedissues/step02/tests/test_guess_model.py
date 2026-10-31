import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from guess_model import GuessModel, PREDICTORS, TARGETS, normalize_config

def row(i, a=0.1, b=0.8, criterion="squared_error"):
    return {"MRE-": a+i*.01, "ACC+": b-i*.01, "PRED40+": 0.0, "N_estimators": 10+i,
            "criterion": criterion, "Min_sample_leaves": 1, "Min_impurity_decrease": i*.1,
            "Max_depth": 3}

class Tests(unittest.TestCase):
    def test_grouping_and_exclusion(self):
        r=row(0); s=dict(r); s["PRED40+"]=99
        self.assertEqual(normalize_config(r, strict=False), normalize_config(s, strict=False))
        self.assertEqual(len(GuessModel([row(i) for i in range(5)]).train_keys), 4)
    def test_reproducible_and_prediction(self):
        rows=[row(i, i, 10-i) for i in range(6)]
        a,b=GuessModel(rows),GuessModel(rows)
        config={x: rows[0][x] for x in PREDICTORS}
        self.assertEqual(a.train_keys,b.train_keys); self.assertEqual(a.predict(config),b.predict(config))
    def test_missing_unseen_and_schema(self):
        rows=[row(i) for i in range(4)]; m=GuessModel(rows)
        c={x: rows[0][x] for x in PREDICTORS}
        c["criterion"]="new"; self.assertTrue(all(math.isfinite(v) for v in m.predict(c).values()))
        c["Max_depth"]=None; self.assertTrue(m.predict(c))
        with self.assertRaises(ValueError): m.predict({x:1 for x in PREDICTORS[:-1]})
    def test_targets_and_data_rules(self):
        z=row(0,0,0); self.assertEqual(z["MRE-"],0); self.assertEqual(z["ACC+"],0)
        with self.assertRaises(ValueError): GuessModel([row(0),row(0)])
        bad=row(1); bad["MRE-"]=None
        with self.assertRaises(ValueError): GuessModel([row(0),bad])

if __name__ == "__main__": unittest.main()
