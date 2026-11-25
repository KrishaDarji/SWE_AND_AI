"""Predict Performance- using encoder settings and a reproducible holdout."""

import argparse
import json
import math
from pathlib import Path
import platform

import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from remember import OBJECTIVE, load_measurements


SEED = 42
TEST_SIZE = 0.2
MODEL_SETTINGS = dict(n_estimators=200, criterion="squared_error", max_depth=None,
                      min_samples_split=2, min_samples_leaf=1, max_features=1.0,
                      bootstrap=True, random_state=SEED, n_jobs=1)


def evaluate(data):
    """Fit once on training rows; repeated configurations stay in one partition."""
    features = [list(row[:-1]) for row in data.rows]
    targets = [row[-1] for row in data.rows]
    identities = {key: index for index, key in enumerate(sorted(set(map(tuple, features))))}
    groups = [identities[tuple(row)] for row in features]
    if len(identities) < 2:
        raise ValueError("Evaluation needs at least two distinct configurations")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train, test = next(splitter.split(features, groups=groups))
    x_train = [features[i] for i in train]
    y_train = [targets[i] for i in train]
    x_test = [features[i] for i in test]
    y_test = [targets[i] for i in test]
    model = RandomForestRegressor(**MODEL_SETTINGS).fit(x_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
    predicted = model.predict(x_test)
    baseline_predicted = baseline.predict(x_test)
    report = {
        "objective": OBJECTIVE, "direction": "minimize", "units": None,
        "features": list(data.settings),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "split": {"method": "GroupShuffleSplit", "seed": SEED,
                  "test_fraction_of_unique_configurations": TEST_SIZE,
                  "train_count": len(train), "test_count": len(test),
                  "train_data_rows": [int(i) + 1 for i in train],
                  "test_data_rows": [int(i) + 1 for i in test]},
        "model": {"class": "RandomForestRegressor", "settings": dict(MODEL_SETTINGS)},
        "preprocessing": "None; finite numeric settings in CSV column order",
        "model_mae": float(mean_absolute_error(y_test, predicted)),
        "baseline": {"class": "DummyRegressor", "strategy": "mean",
                     "training_mean": float(baseline.constant_[0, 0])},
        "baseline_mae": float(mean_absolute_error(y_test, baseline_predicted)),
        "held_out_predictions": [
            {"data_row": int(i) + 1, "actual": actual, "predicted": float(prediction)}
            for i, actual, prediction in zip(test, y_test, predicted)
        ],
        "limitations": [
            "One fixed holdout; no hyperparameter search or uncertainty calibration.",
            "Random configuration holdout does not measure transfer to new videos or hardware.",
            "Units and flag semantics are unspecified; no causal interpretation.",
            "Constant settings have no measured effect; extrapolation is unsupported.",
            "Predictions retain the training-only fit; evaluation rows are never used to fit.",
            "Per-setting domain checks do not establish validity of combined encoder settings.",
        ],
    }
    return model, report


def predict_configuration(model, data, report, configuration):
    """Require every setting, no target, and values observed in training only."""
    if not isinstance(configuration, dict) or set(configuration) != set(data.settings):
        raise ValueError("Configuration must contain exactly the encoder settings; exclude Performance-")
    train_rows = [data.rows[i - 1] for i in report["split"]["train_data_rows"]]
    values = []
    for index, name in enumerate(data.settings):
        value = configuration[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Setting {name} must be a finite number")
        if value not in {row[index] for row in train_rows}:
            raise ValueError(f"Setting {name} is outside the observed training domain")
        values.append(float(value))
    return {"settings": dict(zip(data.settings, values)),
            "predicted_Performance-": float(model.predict([values])[0]),
            "units": None, "fit": "training partition only"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="JSON object with encoder settings only")
    args = parser.parse_args(argv)
    try:
        data = load_measurements(args.dataset)
        model, report = evaluate(data)
        if args.config:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            report["configuration_prediction"] = predict_configuration(model, data, report, config)
        target = args.output_dir / "evaluation.json"
        protected = [args.dataset.resolve()]
        if args.config:
            protected.append(args.config.resolve())
        if target.resolve() in protected:
            raise ValueError("Output must not overwrite an input")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({key: report[key] for key in ("model_mae", "baseline_mae", "split")},
                     sort_keys=True))
    if "configuration_prediction" in report:
        print(json.dumps(report["configuration_prediction"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
