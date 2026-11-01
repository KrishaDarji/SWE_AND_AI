import csv, os, tempfile, unittest
from source.analyzer import spot

H = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"
def make(rows):
    f = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False); csv.writer(f).writerows([H.rstrip("\n").split(",")] + rows); f.close(); return f.name
def r(m, a, p, n=1): return [m, a, p, n, "squared_error", 1, 0, 3]

class SpotTests(unittest.TestCase):
    def test_high_low_all_outcomes_and_retention(self):
        path = make([r(1, 10, 100, 1), r(2, 20, 200, 2), r(3, 30, 300, 3), r(100, -100, 1000, 4)])
        try:
            x = spot(path, threshold=1.0); self.assertEqual(x["outcomes"]["MRE-"]["flag_count"], 2); self.assertEqual(x["outcomes"]["ACC+"]["flag_count"], 2); self.assertEqual(x["outcomes"]["PRED40+"]["flag_count"], 2)
            self.assertEqual(x["unique_flagged_row_count"], 3); retained = next(z for z in x["flagged_rows"] if z["position"] == 5); self.assertEqual(retained["configuration"]["N_estimators"], 4.0)
        finally: os.unlink(path)
    def test_boundary_and_independent_reference(self):
        path = make([r(0, 1, 2), r(1, 1, 2), r(2, 1, 2), r(3, 1, 2)])
        try:
            x = spot(path, threshold=1.02); self.assertEqual(x["outcomes"]["MRE-"]["flag_count"], 0); self.assertEqual(x["outcomes"]["ACC+"]["reference_count"], 4); self.assertEqual(x["outcomes"]["ACC+"]["undefined_reason"], "zero_MAD_reference")
        finally: os.unlink(path)
    def test_zero_mad_short_missing_nonfinite_reconcile(self):
        path = make([["?", 1, 2, 1, "c", 1, 0, 1], ["nan", 2, 3, 2, "c", 1, 0, 1]])
        try:
            x = spot(path); s = x["outcomes"]["MRE-"]; self.assertEqual(s["status_counts"], {"invalid_or_missing": 2, "undefined_score": 0, "ordinary_nonflag": 0, "flagged": 0}); self.assertEqual(x["outcomes"]["ACC+"]["undefined_reason"], "insufficient_finite_reference")
        finally: os.unlink(path)
    def test_empty_and_malformed_position(self):
        path = make([[1, 2, 3, 1, "c", 1, 0, 1]])
        try:
            x = spot(path); self.assertEqual(x["outcomes"]["MRE-"]["reference_count"], 1); self.assertEqual(x["outcomes"]["MRE-"]["status_counts"]["undefined_score"], 1)
        finally: os.unlink(path)
if __name__ == "__main__": unittest.main()
