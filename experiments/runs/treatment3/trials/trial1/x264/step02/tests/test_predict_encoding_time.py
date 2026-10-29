import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("predict_encoding_time", HERE.parents[1] / "source" / "predict_encoding_time.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DATASET = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/config/X264_AllMeasurements.csv")

class PredictorTests(unittest.TestCase):
    def test_loads_expected_shape(self):
        names, matrix, target = module.load_measurements(str(DATASET))
        self.assertEqual(len(names), 16)
        self.assertEqual((len(matrix), len(target)), (1152, 1152))

    def test_prediction_and_validation(self):
        model, names = module.fit_predictor(str(DATASET))
        prediction = module.predict_configuration(model, names, {name: 0 for name in names})
        self.assertGreater(prediction, 0)
        with self.assertRaises(ValueError):
            module.predict_configuration(model, names, {names[0]: 0})

    def test_evaluation_is_deterministic(self):
        self.assertAlmostEqual(module.evaluate(str(DATASET)), module.evaluate(str(DATASET)), places=10)

if __name__ == "__main__":
    unittest.main()
