"""Holdout isolation, prediction validation, and CLI integration tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

STEP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP / "source"))
import guess
from remember import Measurements, load_measurements


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Measurements(("a", "b"), tuple(
            (float(i % 2), float(i), float(i * i + 2)) for i in range(24)))
        cls.model, cls.report = guess.evaluate(cls.data)

    def test_repeatability_and_independent_mae_baseline(self):
        model, report = guess.evaluate(self.data)
        self.assertEqual(report, self.report)
        self.assertEqual(model.n_features_in_, 2)
        train = report["split"]["train_data_rows"]
        test = report["split"]["test_data_rows"]
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train + test), set(range(1, 25)))
        mean = sum(self.data.rows[i - 1][-1] for i in train) / len(train)
        self.assertAlmostEqual(report["baseline"]["training_mean"], mean)
        baseline_mae = sum(abs(self.data.rows[i - 1][-1] - mean) for i in test) / len(test)
        self.assertAlmostEqual(report["baseline_mae"], baseline_mae)
        model_mae = sum(abs(p["actual"] - p["predicted"])
                        for p in report["held_out_predictions"]) / len(test)
        self.assertAlmostEqual(report["model_mae"], model_mae)

    def test_evaluation_targets_do_not_affect_fit(self):
        held_out = set(self.report["split"]["test_data_rows"])
        changed = Measurements(self.data.settings, tuple(
            (*row[:-1], row[-1] + 10000 if i in held_out else row[-1])
            for i, row in enumerate(self.data.rows, 1)))
        model, report = guess.evaluate(changed)
        self.assertEqual(self.report["split"], report["split"])
        self.assertEqual([p["predicted"] for p in self.report["held_out_predictions"]],
                         [p["predicted"] for p in report["held_out_predictions"]])
        self.assertEqual(self.report["baseline"], report["baseline"])
        self.assertNotEqual(self.report["model_mae"], report["model_mae"])

    def test_repeated_configurations_are_grouped(self):
        repeated = Measurements(self.data.settings, self.data.rows + self.data.rows)
        _, report = guess.evaluate(repeated)
        partitions = [{repeated.rows[i - 1][:-1] for i in report["split"][key]}
                      for key in ("train_data_rows", "test_data_rows")]
        self.assertFalse(partitions[0] & partitions[1])
        with self.assertRaises(ValueError):
            guess.evaluate(Measurements(("a",), ((1., 2.), (1., 3.))))

    def test_prediction_schema_domain_and_column_order(self):
        row = self.data.rows[self.report["split"]["train_data_rows"][0] - 1]
        config = {"b": row[1], "a": row[0]}
        prediction = guess.predict_configuration(self.model, self.data, self.report, config)
        self.assertEqual(prediction["predicted_Performance-"], self.model.predict([row[:-1]])[0])
        for bad in ({"a": 1}, {**config, "Performance-": 1},
                    {**config, "a": float("nan")}, {**config, "a": "1"},
                    {**config, "a": True}, {**config, "a": 9}, []):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                guess.predict_configuration(self.model, self.data, self.report, bad)

    def test_evaluation_only_setting_value_is_rejected(self):
        train_values = {self.data.rows[i - 1][1]
                        for i in self.report["split"]["train_data_rows"]}
        heldout_only = {row[1] for row in self.data.rows} - train_values
        self.assertTrue(heldout_only)
        with self.assertRaisesRegex(ValueError, "training domain"):
            guess.predict_configuration(self.model, self.data, self.report,
                                        {"a": 0, "b": next(iter(heldout_only))})

    def test_actual_dataset_cli_prediction_and_input_preservation(self):
        dataset = Path(os.environ["X264_DATASET"])
        before = dataset.read_bytes()
        data = load_measurements(dataset)
        with tempfile.TemporaryDirectory(dir=STEP / "tests") as tmp:
            directory = Path(tmp)
            config = directory / "config.json"
            config.write_text(json.dumps(dict(zip(data.settings, data.rows[0][:-1]))))
            process = subprocess.run([sys.executable, "-B", str(STEP / "source" / "guess.py"),
                                      str(dataset), "--config", str(config),
                                      "--output-dir", str(directory)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads((directory / "evaluation.json").read_text())
            self.assertEqual((report["split"]["train_count"], report["split"]["test_count"]),
                             (921, 231))
            self.assertEqual(report["features"], list(data.settings))
            self.assertNotIn("Performance-", report["features"])
            self.assertIn("configuration_prediction", report)
            self.assertIsNone(report["units"])
        self.assertEqual(dataset.read_bytes(), before)

    def test_cli_invalid_config_preserves_input(self):
        with tempfile.TemporaryDirectory(dir=STEP / "tests") as tmp:
            directory = Path(tmp)
            dataset = directory / "data.csv"
            dataset.write_text("a,Performance-\n0,2\n1,3\n")
            config = directory / "bad.json"
            config.write_text('{"Performance-": 1}')
            process = subprocess.run([sys.executable, "-B", str(STEP / "source" / "guess.py"),
                                      str(dataset), "--config", str(config),
                                      "--output-dir", str(directory)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2)
            self.assertIn("exactly the encoder settings", process.stderr)
            self.assertFalse((directory / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
