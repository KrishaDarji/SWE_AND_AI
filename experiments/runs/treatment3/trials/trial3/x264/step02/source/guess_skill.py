#!/usr/bin/env python3
"""Independent x264 encoding-time regression skill."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_TARGET = "Performance-"


class DatasetError(ValueError):
    pass


def load_dataset(csv_path, target=DEFAULT_TARGET):
    """Load a numeric CSV and return (feature_names, X, y, validation_notes)."""
    path = Path(csv_path)
    if not path.is_file():
        raise DatasetError("dataset does not exist: %s" % path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or target not in reader.fieldnames:
            raise DatasetError("missing target column %r" % target)
        feature_names = [name for name in reader.fieldnames if name != target]
        rows = list(reader)
    if not rows or not feature_names:
        raise DatasetError("dataset must contain rows and features")
    X, y = [], []
    for row_number, row in enumerate(rows, start=2):
        if any(row.get(name, "").strip() == "" for name in reader.fieldnames):
            raise DatasetError("blank value at CSV row %d" % row_number)
        try:
            values = [float(row[name]) for name in feature_names]
            target_value = float(row[target])
        except (TypeError, ValueError) as exc:
            raise DatasetError("non-numeric value at CSV row %d" % row_number) from exc
        if not all(math.isfinite(v) for v in values + [target_value]):
            raise DatasetError("non-finite value at CSV row %d" % row_number)
        X.append(values)
        y.append(target_value)
    if len(X) < 10:
        raise DatasetError("at least 10 rows are required")
    return feature_names, X, y


class X264TimeGuesser:
    """Fit and query an ExtraTrees ensemble for encoding-time predictions."""

    def __init__(self, random_state=20261029, n_estimators=300):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = None
        self.active_indices = None

    def fit(self, feature_names, X, y):
        self.feature_names = list(feature_names)
        self.active_indices = [i for i in range(len(feature_names))
                               if len({row[i] for row in X}) > 1]
        if not self.active_indices:
            raise DatasetError("all configuration features are constant")
        reduced = [[row[i] for i in self.active_indices] for row in X]
        self.model = ExtraTreesRegressor(
            n_estimators=self.n_estimators, random_state=self.random_state,
            min_samples_leaf=2, n_jobs=1)
        self.model.fit(reduced, y)
        return self

    def predict(self, configuration):
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        if set(configuration) != set(self.feature_names):
            missing = sorted(set(self.feature_names) - set(configuration))
            extra = sorted(set(configuration) - set(self.feature_names))
            raise ValueError("configuration keys mismatch; missing=%s extra=%s" % (missing, extra))
        values = [float(configuration[name]) for name in self.feature_names]
        reduced = [[values[i] for i in self.active_indices]]
        tree_values = [tree.predict(reduced)[0] for tree in self.model.estimators_]
        prediction = sum(tree_values) / len(tree_values)
        spread = math.sqrt(sum((v - prediction) ** 2 for v in tree_values) / len(tree_values))
        return {"prediction": prediction, "uncertainty_std": spread}


def evaluate(csv_path, target=DEFAULT_TARGET, test_size=0.2, random_state=20261029):
    names, X, y = load_dataset(csv_path, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    model = X264TimeGuesser(random_state=random_state).fit(names, X_train, y_train)
    predictions = model.model.predict([[row[i] for i in model.active_indices] for row in X_test])
    mae = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    nonzero = [abs(v) for v in y_test if v != 0]
    mape = sum(abs(a - b) / abs(b) for a, b in zip(predictions, y_test) if b != 0) / len(nonzero) * 100
    metrics = {"rows": len(X), "features": len(names), "active_features": len(model.active_indices),
               "train_rows": len(X_train), "test_rows": len(X_test), "mae": mae,
               "rmse": rmse, "r2": r2_score(y_test, predictions), "mape_percent": mape}
    return metrics, model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict x264 encoding time")
    parser.add_argument("--csv", required=True, help="measurement CSV")
    parser.add_argument("--evaluate", action="store_true", help="report held-out test metrics")
    parser.add_argument("--config", nargs="*", help="feature=value pairs for a prediction")
    args = parser.parse_args(argv)
    metrics, model = evaluate(args.csv)
    if args.evaluate or not args.config:
        print(json.dumps(metrics, sort_keys=True))
    if args.config:
        config = dict(item.split("=", 1) for item in args.config)
        print(json.dumps(model.predict(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
