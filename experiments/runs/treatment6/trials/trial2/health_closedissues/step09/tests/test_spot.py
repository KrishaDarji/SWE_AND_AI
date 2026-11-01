import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from spot import screen

HEADER=["MRE-","ACC+","PRED40+","N_estimators","criterion","Min_sample_leaves","Min_impurity_decrease","Max_depth"]
def make(rows):
    f=tempfile.NamedTemporaryFile(mode="w", newline="", delete=False); w=csv.writer(f); w.writerow(HEADER); w.writerows(rows); f.close(); return f.name

class SpotTests(unittest.TestCase):
    def test_independent_flags_positions_and_config(self):
        p=make([[1,10,5,1,"x",2,0,3],[2,20,6,2,"y",3,1,4],[3,30,7,3,"z",4,2,5],[100,40,8,4,"q",5,3,6]])
        try:
            r=screen(p, threshold=3.5); self.assertEqual(r["outcomes"]["MRE-"]["reference_count"],4); self.assertEqual(r["outcomes"]["MRE-"]["statuses"]["flag"],1); self.assertEqual(r["flags"][0]["position"],5); self.assertEqual(r["flags"][0]["configuration"]["Max_depth"],"6")
        finally: os.unlink(p)
    def test_missing_nonfinite_and_zero_mad(self):
        p=make([[1,1,1,1,"x",1,0,1],["nan",2,2,2,"x",1,0,1],[1,"inf",3,3,"x",1,0,1],[1,3,4,4,"x",1,0,1]])
        try:
            r=screen(p); self.assertFalse(r["outcomes"]["MRE-"]["defined"]); self.assertEqual(r["outcomes"]["MRE-"]["statuses"]["invalid_or_missing"],1); self.assertEqual(r["outcomes"]["MRE-"]["statuses"]["undefined_score"],3); self.assertEqual(r["outcomes"]["ACC+"]["statuses"]["invalid_or_missing"],1)
        finally: os.unlink(p)
    def test_short_and_boundary(self):
        p=make([[1,1,1,1,"x",1,0,1],[2,2,2,2,"x",1,0,1]])
        try: self.assertFalse(screen(p)["outcomes"]["PRED40+"]["defined"])
        finally: os.unlink(p)
if __name__ == "__main__": unittest.main()
