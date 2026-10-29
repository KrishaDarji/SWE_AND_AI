import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
DATASET = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(SOURCE))
from effort_estimator import estimate, fit_and_score, read_dataset  # noqa: E402


class EffortEstimatorTests(unittest.TestCase):
    def test_reads_numeric_and_categorical_features(self):
        rows, targets = read_dataset(DATASET)
        self.assertEqual(len(rows), 93)
        self.assertEqual(len(targets), 93)
        self.assertIsInstance(rows[0]["YearX"], float)
        self.assertEqual(rows[0]["prec"], "h")

    def test_fit_scores_and_predicts_positive_effort(self):
        model, metrics = fit_and_score(DATASET)
        self.assertGreater(metrics["test_rows"], 0)
        self.assertGreaterEqual(metrics["mae"], 0)
        rows, _ = read_dataset(DATASET)
        prediction = model.predict([rows[0]])[0]
        self.assertGreater(prediction, 0)

    def test_full_data_estimate_is_finite(self):
        rows, _ = read_dataset(DATASET)
        prediction = estimate(DATASET, rows[0])
        self.assertTrue(prediction > 0)

    def test_cli_returns_json(self):
        rows, _ = read_dataset(DATASET)
        result = subprocess.run(
            [sys.executable, str(SOURCE / "effort_estimator.py"), str(DATASET), json.dumps(rows[0])],
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        self.assertIn("estimated_effort_person_months", payload)


if __name__ == "__main__":
    unittest.main()
