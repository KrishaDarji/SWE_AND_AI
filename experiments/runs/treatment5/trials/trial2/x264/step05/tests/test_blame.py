import csv, os, tempfile, unittest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),"..","source"))
from blame import analyze

class TestBlame(unittest.TestCase):
    def data(self, rows):
        f=tempfile.NamedTemporaryFile("w",newline="",delete=False); w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); f.close(); self.addCleanup(lambda:os.unlink(f.name)); return f.name
    def test_known_constant_multivalue_missing_and_order(self):
        rows=[{"fast":"0","constant":"x","multi":"a","correlated":"0","Performance-":"10"},{"fast":"1","constant":"x","multi":"b","correlated":"1","Performance-":"5"},{"fast":"0","constant":"x","multi":"a","correlated":"0","Performance-":"11"},{"fast":"1","constant":"x","multi":"b","correlated":"1","Performance-":"4"},{"fast":"","constant":"x","multi":"c","correlated":"1","Performance-":"?"}]
        r=analyze(self.data(rows)); d={x["setting"]:x for x in r["ranking"]}
        self.assertEqual(r["rows"],5); self.assertEqual(d["constant"]["score"],0); self.assertEqual(d["multi"]["groups"],2); self.assertEqual(d["fast"]["direction"],"decreasing"); self.assertEqual(d["correlated"]["score"],d["fast"]["score"])
        self.assertEqual([x["setting"] for x in r["ranking"]],sorted([x["setting"] for x in r["ranking"]],key=lambda n:(-d[n]["score"],n)))
    def test_nonfinite(self):
        rows=[{"x":"0","Performance-":"nan"},{"x":"1","Performance-":"3"}]
        self.assertEqual(analyze(self.data(rows))["outcome_valid"],1)
    def test_numeric_directions_and_nonmonotonic(self):
        for vals, expected in [([("0","1"),("1","2"),("2","3")],"increasing"),([("0","3"),("1","2"),("2","1")],"decreasing"),([("0","1"),("1","3"),("2","2")],"nonmonotonic")]:
            rows=[{"x":x,"Performance-":y} for x,y in vals]; item=analyze(self.data(rows))["ranking"][0]; self.assertEqual(item["direction"],expected)
    def test_unordered_categories_and_tied_means(self):
        rows=[{"x":"beta","Performance-":"1"},{"x":"alpha","Performance-":"3"},{"x":"gamma","Performance-":"2"}]
        item=analyze(self.data(rows))["ranking"][0]; self.assertEqual(item["direction"],"unordered-categorical"); self.assertEqual(item["lowest_mean_groups"],["beta"]); self.assertEqual(item["highest_mean_groups"],["alpha"])
        rows=[{"x":"beta","Performance-":"2"},{"x":"alpha","Performance-":"2"}]
        item=analyze(self.data(rows))["ranking"][0]; self.assertEqual(item["direction"],"tied"); self.assertEqual(item["lowest_mean_groups"],["alpha","beta"])
if __name__=="__main__": unittest.main()
