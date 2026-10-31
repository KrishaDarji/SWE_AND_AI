import math, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from blame import association, analyze, read_rows

class BlameTests(unittest.TestCase):
    def test_known_association_and_tie_order(self):
        rows = [{"A":" high ","B":"x","Kloc+":"1","Effort-":"10","Defects-":"1"},
                {"A":"high","B":"x","Kloc+":"2","Effort-":"12","Defects-":"2"},
                {"A":"low","B":"y","Kloc+":"3","Effort-":"2","Defects-":"3"},
                {"A":"low","B":"y","Kloc+":"4","Effort-":"4","Defects-":"4"}]
        r = association(rows, "A")
        self.assertAlmostEqual(r["score"], 16 / 17, places=12)
        self.assertEqual(r["groups"]["high"], [10.0, 12.0])
        self.assertEqual(r["high"], "high")
    def test_excluded_and_categorical(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            f.write("idX,A,Kloc+,Effort-,Defects-,MonthsX\n1,h,1,2,3,4\n2,l,2,4,5,6\n")
            name = f.name
        try:
            rows, descriptors = read_rows(name)
            self.assertEqual(descriptors, ["A"])
            self.assertEqual(len(analyze(name)), 1)
        finally: os.unlink(name)
    def test_constants_missing_small_and_nonfinite(self):
        rows = [{"A":"x","Effort-":"?"},{"A":"x","Effort-":"nan"},
                {"A":"x","Effort-":"3"}]
        self.assertEqual(association(rows, "A")["score"], 0.0)
        rows[0]["A"] = "?"; rows[0]["Effort-"] = "4"
        rows[1]["A"] = "y"; rows[1]["Effort-"] = "5"
        self.assertEqual(association(rows, "A")["overall"]["n"], 2)
    def test_deterministic_order(self):
        rows = [{"A":"x","B":"u","Effort-":"1"},{"A":"y","B":"v","Effort-":"1"}]
        self.assertEqual([x["descriptor"] for x in sorted([association(rows,"B"), association(rows,"A")], key=lambda x:(-x["score"],x["descriptor"]))], ["A","B"])

if __name__ == "__main__": unittest.main()
