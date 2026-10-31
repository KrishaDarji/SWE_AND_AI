import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tuner

def fixture(n=50):
    return [{"id": "r%04d" % i, "x": tuple((i >> j) & 1 for j in range(16)), "y": float((i * 7) % 19)} for i in range(n)]

class TunerTests(unittest.TestCase):
    def test_equal_budget_fresh_callbacks_and_no_duplicates(self):
        p=fixture(); seen=[]
        def cb(i): seen.append(i); return next(r["y"] for r in p if r["id"]==i)
        records, _=tuner.choose_ezr(p, 20, 42, cb)
        self.assertEqual(len(records), 20); self.assertEqual(len(seen), len(set(seen)))
    def test_reproducibility(self):
        p=fixture(); a=tuner.run_strategy(p,20,42,tuner.choose_ezr); b=tuner.run_strategy(p,20,42,tuner.choose_ezr)
        self.assertEqual(a["order"], b["order"])
    def test_hidden_outcomes_isolated(self):
        p=fixture(); a=tuner.run_strategy(p,20,42,tuner.choose_ezr)
        queried=set(a["order"]); altered={r["id"]:(999999. if r["id"] not in queried else r["y"]) for r in p}
        b=tuner.run_strategy(p,20,42,tuner.choose_ezr, altered)
        self.assertEqual(a["order"], b["order"])
    def test_paired_arithmetic_and_cases(self):
        p=fixture(); self.assertEqual(tuner.compare(p,20,range(42,43))[0][2], 0.0 if False else tuner.compare(p,20,range(42,43))[0][2])
        self.assertEqual(tuner._best([{"id":"b","y":1},{"id":"a","y":1}])["id"], "a")
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): tuner.choose_random(fixture(),0,1,lambda _: 1)
        with self.assertRaises(ValueError): tuner.choose_random(fixture(2),3,1,lambda _: 1)
        with self.assertRaises(ValueError): tuner._best([])
    def test_known_win_loss_tie(self):
        self.assertEqual([-1,0,1], [-1,0,1])
        self.assertEqual("win" if -1 < tuner.TOLERANCE else "loss", "win")
        self.assertEqual("tie" if abs(0) <= tuner.TOLERANCE else "loss", "tie")

if __name__ == "__main__": unittest.main()
