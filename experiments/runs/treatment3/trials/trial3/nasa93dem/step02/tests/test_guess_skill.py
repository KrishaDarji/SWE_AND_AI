import unittest
from pathlib import Path
from source.guess_skill import DatasetError, EffortGuesser, load_dataset

ROOT = Path(__file__).resolve().parents[8]
DATASET = ROOT / "moot/optimize/process/nasa93dem.csv"


class GuessTests(unittest.TestCase):
    def test_loads_target_and_excludes_leakage(self):
        names, X, y = load_dataset(DATASET)
        self.assertEqual(len(X), 93)
        self.assertIn("Kloc+", names)
        self.assertNotIn("Defects-", names)
        self.assertNotIn("MonthsX", names)
        self.assertEqual(len(y), 93)

    def test_prediction_has_uncertainty(self):
        names, X, y = load_dataset(DATASET)
        model = EffortGuesser(n_estimators=12).fit(names, X[:70], y[:70])
        result = model.predict(X[70])
        self.assertGreater(result["prediction"], 0)
        self.assertGreaterEqual(result["uncertainty_std"], 0)

    def test_missing_file_rejected(self):
        with self.assertRaises(DatasetError):
            load_dataset("does-not-exist.csv")


if __name__ == "__main__":
    unittest.main()
