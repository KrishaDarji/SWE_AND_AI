import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source" / "predictor.py"
DATASET = Path("moot/optimize/hpo/Health-ClosedIssues0000.csv")
sys.path.insert(0, str(SOURCE.parent))
from predictor import ConfigurationPredictor, load  # noqa: E402


class PredictorTests(unittest.TestCase):
    def test_predictions_are_bounded(self):
        rows = load(DATASET)
        result = ConfigurationPredictor(rows).predict(rows[0])
        self.assertTrue(0 <= result["predicted_error"] <= 1)
        self.assertTrue(0 <= result["predicted_accuracy"] <= 1)

    def test_cli_emits_json(self):
        run = subprocess.run([sys.executable, str(SOURCE), str(DATASET)], capture_output=True, text=True, check=True)
        result = json.loads(run.stdout)
        self.assertIn("predicted_error", result)
        self.assertIn("predicted_accuracy", result)


if __name__ == "__main__":
    unittest.main()
