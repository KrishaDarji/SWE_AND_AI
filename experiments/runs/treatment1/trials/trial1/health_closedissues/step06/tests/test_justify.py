import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
import justify

class JustifyTests(unittest.TestCase):
    def test_analysis_recommends_best_configuration(self):
        rows = []
        for score, criterion in ((1, "good"), (.9, "good"), (.2, "bad")):
            rows.append({"MRE-": .1, "ACC+": score, "PRED40+": .1,
                         "N_estimators": "100", "criterion": criterion,
                         "Min_sample_leaves": "2", "Min_impurity_decrease": "0.1",
                         "Max_depth": "5", "score": score})
        result = justify.analyze(rows)
        self.assertEqual(result["best"]["criterion"], "good")
        self.assertEqual(result["recommendation"]["criterion"], "good")

    def test_load_skips_malformed_rows(self):
        path = pathlib.Path(__file__).parent / "fixture.csv"
        path.write_text("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n.1,.9,.8,10,x,1,0,5\nnope,.2,.3,10,y,1,0,5\n")
        try:
            self.assertEqual(len(justify.load_records(path)), 1)
        finally:
            path.unlink()

if __name__ == "__main__":
    unittest.main()
