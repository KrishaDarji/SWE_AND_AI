import tempfile
import unittest
from pathlib import Path
from source.nasa93_summary import rank_effort_associations


class BlameTests(unittest.TestCase):
    def path(self, text):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(text); f.close(); p = Path(f.name)
        self.addCleanup(lambda: p.unlink(missing_ok=True)); return p

    def test_known_association_exclusions_and_order(self):
        p = self.path("idX,signal,constant,Kloc+,Effort-,Defects-,MonthsX\n" +
                      "1,a,x,1,10,1,2\n2,a,x,1,12,1,2\n3,b,x,1,30,1,2\n4,b,x,1,32,1,2\n")
        result = rank_effort_associations(p)
        self.assertEqual(result["predictors"], ["signal", "constant"])
        self.assertEqual([x["descriptor"] for x in result["ranking"]], ["signal", "constant"])
        self.assertEqual(result["ranking"][0]["groups"][0]["n"], 2)
        self.assertEqual(result["ranking"][0]["highest_mean_level"], "b")
        self.assertEqual(result["ranking"][1]["score_eta_squared"], 0.0)

    def test_missing_is_category_and_ties_are_deterministic(self):
        p = self.path("a,b,Effort-\n?,z,1\nq,z,3\n?,y,5\nq,y,7\n")
        first = rank_effort_associations(p); second = rank_effort_associations(p)
        self.assertEqual(first, second)
        self.assertTrue(any("<MISSING>" in [g["level"] for g in item["groups"]]
                            for item in first["ranking"]))


if __name__ == "__main__": unittest.main()
