import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("certify", ROOT / "source" / "certify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CertificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT.parents[6] / "moot" / "optimize" / "process" / "nasa93dem.csv"
        cls.rows = module.read_projects(cls.data)

    def test_dataset_and_feature_shapes(self):
        self.assertEqual(len(self.rows), 93)
        self.assertGreater(len(module.full_features(self.rows[0])), len(module.size_features(self.rows[0])))

    def test_models_predict_positive_finite_effort(self):
        for kind in ("size", "full"):
            prediction = module.predict(module.fit(self.rows, kind), self.rows[0])
            self.assertTrue(math.isfinite(prediction) and prediction > 0)

    def test_evaluation_has_one_error_per_project_per_fold(self):
        errors = module.evaluate(self.rows)
        self.assertEqual(len(errors["size"]), 930)
        self.assertEqual(len(errors["full"]), 930)
        self.assertTrue(all(math.isfinite(value) and value >= 0 for value in errors["full"]))

    def test_certification_is_reproducible_and_complete(self):
        result = module.certify(self.data)
        self.assertEqual(result["projects"], 93)
        self.assertTrue(0 <= result["p_value"] <= 1)
        self.assertLessEqual(result["ci_low"], result["ci_high"])
        self.assertIn(result["winner"], ("size", "full"))


if __name__ == "__main__":
    unittest.main()
