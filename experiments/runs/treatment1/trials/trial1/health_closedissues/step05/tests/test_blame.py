import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import blame

class BlameTests(unittest.TestCase):
    def test_numeric_setting_with_bad_direction_is_reported(self):
        rows = [{"MRE-": str(x), "ACC+": str(1-x), "PRED40+": "0.5", "N_estimators": str(x), "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "5"} for x in (1, 2, 3, 4)]
        report = blame.analyze(rows)
        self.assertIn("N_estimators", report)
        self.assertIn("MRE r=1.000", report)
        self.assertIn("ACC r=-1.000", report)

    def test_categorical_levels_are_compared(self):
        rows = []
        for criterion, error, accuracy in (("good", .1, .9), ("bad", .8, .2)):
            rows.append({"MRE-": str(error), "ACC+": str(accuracy), "PRED40+": "0", "N_estimators": "10", "criterion": criterion, "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "5"})
        self.assertIn("good: MRE=0.100, ACC=0.900, n=1", blame.analyze(rows))

    def test_load_rejects_missing_columns(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as stream:
            csv.writer(stream).writerows([["MRE-"], ["0.2"]])
            path = stream.name
        try:
            with self.assertRaises(ValueError): blame.load_rows(path)
        finally: os.unlink(path)

if __name__ == "__main__": unittest.main()
