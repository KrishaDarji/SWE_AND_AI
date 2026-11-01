import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import summary

class GuessTests(unittest.TestCase):
    def csv(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(text); f.close(); self.addCleanup(lambda: Path(f.name).unlink()); return f.name

    def test_split_groups_and_metrics_are_independent_of_held_out_targets(self):
        text = "a,b,Performance-\n0,x,1\n0,x,3\n1,x,10\n1,x,12\n2,y,20\n3,y,30\n"
        result = summary.evaluate(self.csv(text))
        self.assertEqual(result["train_groups"] + result["test_groups"], 4)
        self.assertEqual(len(set(result["train_indexes"]) & set(result["test_indexes"])), 0)
        self.assertEqual(summary.split_groups(summary.read_runs(self.csv(text))[1], ["a","b"]),
                         summary.split_groups(summary.read_runs(self.csv(text))[1], ["a","b"]))
        self.assertGreaterEqual(result["baseline_mae"], 0)

    def test_validation_constant_unseen_and_exact_fields(self):
        model = summary.NearestMeanModel(["constant", "kind"]).fit(
            [{"constant":"1", "kind":"a"}, {"constant":"1", "kind":"b"}], [0, 2])
        self.assertEqual(model.predict_one({"constant":"1", "kind":"z"}), 1)
        with self.assertRaises(ValueError): model.predict_one({"constant":"1"})
        with self.assertRaises(ValueError): model.predict_one({"constant":"1", "kind":"a", "extra":"x"})
        with self.assertRaises(ValueError): model.predict_one({"constant":"bad", "kind":"a"})

    def test_zero_target_and_insufficient_groups(self):
        result = summary.evaluate(self.csv("a,Performance-\n0,0\n1,2\n"))
        self.assertEqual(result["train_rows"] + result["test_rows"], 2)
        with self.assertRaises(ValueError): summary.split_groups([{"a":"0"}], ["a"])

if __name__ == "__main__": unittest.main()
