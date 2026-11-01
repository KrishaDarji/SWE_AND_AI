import csv, os, tempfile, unittest
from fix import suggest_fix

HEADER = ["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
def run(rows):
    f = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
    try:
        w=csv.writer(f); w.writerow(HEADER); w.writerows(rows); f.close(); return suggest_fix(f.name)
    finally:
        try: os.unlink(f.name)
        except FileNotFoundError: pass

class FixTests(unittest.TestCase):
    def row(self, m, a, p, n="10", c="gini", l="1", d="0", depth="5"): return [m,a,p,n,c,l,d,depth]
    def test_min_changes_before_magnitude_and_ties(self):
        r=run([self.row(0.5,.5,.5), self.row(.1,.4,.4,n="11"), self.row(.2,.9,.9,n="10",depth="6"), self.row(.2,.8,.9,n="10",depth="7")])
        self.assertEqual(r["selected"]["position"],3); self.assertEqual(r["selected"]["change_count"],1)
    def test_numeric_normalization_and_categorical(self):
        r=run([self.row(.5,.5,.5,n="10.0"), self.row(.4,.6,.6,n="10",c="Gini",depth="6")]); self.assertEqual(r["selected"]["change_count"],1)
    def test_invalid_baseline_empty_and_no_improvement(self):
        self.assertEqual(run([])["status"],"empty"); self.assertEqual(run([["bad"]])["status"],"invalid_baseline"); self.assertEqual(run([self.row(.5,.5,.5),self.row(.5,.9,.9)])["status"],"no_improvement")
    def test_worsening_report_and_positions_after_malformed(self):
        r=run([self.row(.5,.5,.5), [1,2], self.row(.4,.4,.6,n="20")]); self.assertEqual(r["selected"]["position"],4); self.assertLess(r["secondary_differences"]["ACC+"],0); self.assertEqual(r["malformed_count"],1)
    def test_missing_candidate_excluded(self):
        r=run([self.row(.5,.5,.5), self.row(.2,.4,.4,c=""), self.row(.3,.6,.6,n="999")]); self.assertEqual(r["selected"]["position"],4)
if __name__ == "__main__": unittest.main()
