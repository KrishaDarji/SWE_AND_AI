import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from x264_guess import TARGET, X264GuessModel, evaluate, load_dataset


DATASET = Path("moot/optimize/config/X264_AllMeasurements.csv")


class X264GuessTests(unittest.TestCase):
    def test_loads_and_validates_shape(self):
        names, X, y = load_dataset(DATASET)
        self.assertEqual(len(names), 16)
        self.assertEqual((len(X), len(y)), (1152, 1152))
        self.assertEqual(TARGET, "Performance-")

    def test_predict_supports_mapping(self):
        names, X, y = load_dataset(DATASET)
        model = X264GuessModel().fit(names, X, y)
        prediction = model.predict(dict(zip(names, X[0])))
        self.assertTrue(prediction > 0)

    def test_metrics_are_finite(self):
        metrics = evaluate(DATASET)
        self.assertEqual(metrics["test_rows"], 231.0)
        for key in ("mae", "rmse", "r2"):
            self.assertTrue(metrics[key] == metrics[key])

    def test_rejects_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
