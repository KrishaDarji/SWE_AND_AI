import csv
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from health_closedissues import predict, read_rows


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"MRE-": "0.1", "ACC+": "0.8", "PRED40+": "0.5", "N_estimators": "10", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"},
            {"MRE-": "0.9", "ACC+": "0.2", "PRED40+": "0.1", "N_estimators": "20", "criterion": "b", "Min_sample_leaves": "3", "Min_impurity_decrease": "1", "Max_depth": "4"},
        ]
        self.query = {"N_estimators": 10, "criterion": "a", "Min_sample_leaves": 1, "Min_impurity_decrease": 0, "Max_depth": 2}

    def test_exact_configuration_returns_observed_targets(self):
        self.assertEqual(predict(self.rows, self.query, 2), {"MRE-": .1, "ACC+": .8, "PRED40+": .5})

    def test_prediction_is_bounded_by_neighbor_targets(self):
        query = dict(self.query, N_estimators=15, criterion="a", Min_sample_leaves=2, Min_impurity_decrease=.5, Max_depth=3)
        result = predict(self.rows, query, 2)
        self.assertTrue(.1 <= result["MRE-"] <= .9)
        self.assertTrue(.2 <= result["ACC+"] <= .8)

    def test_invalid_neighbor_count_rejected(self):
        with self.assertRaises(ValueError):
            predict(self.rows, self.query, 0)

    def test_csv_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.rows[0])
                writer.writeheader(); writer.writerows(self.rows)
            self.assertEqual(len(read_rows(path)), 2)


if __name__ == "__main__":
    unittest.main()
