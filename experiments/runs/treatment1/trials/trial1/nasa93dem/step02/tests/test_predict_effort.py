import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("predict_effort", ROOT / "source" / "predict_effort.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EffortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT.parents[6] / "moot" / "optimize" / "process" / "nasa93dem.csv"
        cls.rows = module.read_projects(cls.data)

    def test_dataset_loaded(self):
        self.assertEqual(len(self.rows), 93)
        self.assertGreater(float(self.rows[0]["Effort-"]), 0)

    def test_features_are_numeric_and_stable(self):
        values = module.features(self.rows[0])
        self.assertGreater(len(values), 5)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_prediction_is_positive(self):
        estimate = module.predict(module.fit(self.rows), self.rows[0])
        self.assertTrue(math.isfinite(estimate))
        self.assertGreater(estimate, 0)

    def test_leave_one_out_metrics(self):
        mae, mape = module.leave_one_out(self.rows)
        self.assertTrue(math.isfinite(mae) and math.isfinite(mape))
        self.assertGreaterEqual(mae, 0)


if __name__ == "__main__":
    unittest.main()
