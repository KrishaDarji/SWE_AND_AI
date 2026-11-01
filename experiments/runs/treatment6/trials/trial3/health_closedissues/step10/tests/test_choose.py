import csv, math, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from analyzer import choose

FIELDS = ["N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth","MRE-","ACC+","PRED40+"]
def row(n, c="squared_error", leaf=1, imp=0, depth=5, mre=.5, acc=.5, pred=.5):
    return [n,c,leaf,imp,depth,mre,acc,pred]
class ChooseTests(unittest.TestCase):
    def run_rows(self, rows, cap=100):
        with tempfile.NamedTemporaryFile(mode="w", newline="", delete=False) as f:
            w=csv.writer(f); w.writerow(FIELDS); w.writerows(rows); name=f.name
        try: return choose(name, cap)
        finally: os.unlink(name)
    def test_cap_and_counts(self):
        x=self.run_rows([row(100,mre=.2),row(101,mre=.01),row(-1),row(float("nan")),row(50,mre=.3)])
        self.assertEqual(x["recommendation"]["original_position"],2); self.assertEqual(x["over_budget_count"],1); self.assertEqual(x["invalid_count"],2)
        self.assertEqual(x["feasible_count"],2)
    def test_exact_cap_and_invalid_cap(self):
        self.assertEqual(self.run_rows([row(100)])["feasible_count"],1)
        for cap in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError): self.run_rows([row(1)], cap)
    def test_tie_breakers_independently(self):
        self.assertEqual(self.run_rows([row(1,mre=.1,acc=.2),row(2,mre=.1,acc=.3)])["recommendation"]["original_position"],3)
        self.assertEqual(self.run_rows([row(1,mre=.1,acc=.3,pred=.2),row(2,mre=.1,acc=.3,pred=.4)])["recommendation"]["original_position"],3)
        self.assertEqual(self.run_rows([row(1,mre=.1,acc=.3,pred=.4),row(2,mre=.1,acc=.3,pred=.4)])["recommendation"]["original_position"],2)
    def test_missing_settings_outcomes_and_no_candidate(self):
        bad=row(1); bad[1]=""; self.assertEqual(self.run_rows([bad])["status"],"no_feasible_candidate")
        bad=row(1); bad[5]="inf"; x=self.run_rows([bad]); self.assertEqual(x["invalid_count"],1); self.assertIsNone(x["recommendation"])
    def test_over_budget_excellent_cannot_win_and_configuration_retained(self):
        x=self.run_rows([row(101,c="absolute_error",leaf=2,imp=.2,depth=7,mre=0,acc=1,pred=1),row(100,c="squared_error",leaf=3,imp=.3,depth=8,mre=.5,acc=.2,pred=.2)])
        self.assertEqual(x["recommendation"]["original_position"],3)
        self.assertEqual(x["recommendation"]["configuration"]["criterion"],"squared_error")
if __name__ == "__main__": unittest.main()
