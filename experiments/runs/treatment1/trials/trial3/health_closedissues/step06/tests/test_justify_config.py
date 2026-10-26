import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from justify_config import load_rows, recommend


HEADER = "MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n"


class RecommendationTests(unittest.TestCase):
    def make_csv(self, body):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(HEADER + body)
        f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True))
        return f.name

    def test_balanced_score_selects_best_row(self):
        path = self.make_csv("0.1,0.9,0.9,100,squared_error,2,0.1,8\n0.9,0.1,0.1,200,absolute_error,5,1.0,2\n")
        result = recommend(load_rows(path))
        self.assertEqual(result["configuration"]["N_estimators"], 100)
        self.assertEqual(result["dataset_rows"], 2)

    def test_missing_column_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
            f.write("wrong\n1\n")
            f.flush()
            with self.assertRaises(ValueError):
                load_rows(f.name)


if __name__ == "__main__":
    unittest.main()
