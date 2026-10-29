import csv
import tempfile
import unittest
from pathlib import Path

from source.guess_skill import DatasetError, X264TimeGuesser, load_dataset


ROOT = Path(__file__).resolve().parents[8]
DATASET = ROOT / "moot/optimize/config/X264_AllMeasurements.csv"


class GuessSkillTests(unittest.TestCase):
    def test_loads_and_identifies_target(self):
        names, X, y = load_dataset(DATASET)
        self.assertEqual(len(X), 1152)
        self.assertEqual(len(names), 16)
        self.assertEqual(len(y), 1152)

    def test_fit_and_prediction_has_uncertainty(self):
        names, X, y = load_dataset(DATASET)
        model = X264TimeGuesser(n_estimators=20).fit(names, X[:900], y[:900])
        config = dict(zip(names, X[900]))
        result = model.predict(config)
        self.assertGreater(result["prediction"], 0)
        self.assertGreaterEqual(result["uncertainty_std"], 0)

    def test_rejects_non_numeric_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature", "Performance-"])
                writer.writerow(["not-a-number", "1"])
            with self.assertRaises(DatasetError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
