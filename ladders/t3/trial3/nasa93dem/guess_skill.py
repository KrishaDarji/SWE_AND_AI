#!/usr/bin/env python3
"""NASA93DEM software-effort estimation skill."""
import argparse
import csv
import json
import math
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Effort-"
LEAKAGE_COLUMNS = {"Defects-", "MonthsX"}
ID_COLUMNS = {"idX"}


class DatasetError(ValueError):
    pass


def load_dataset(csv_path, target=TARGET):
    path = Path(csv_path)
    if not path.is_file():
        raise DatasetError("dataset does not exist: %s" % path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if target not in fields:
            raise DatasetError("missing target column %r" % target)
        rows = list(reader)
    if not rows:
        raise DatasetError("dataset has no rows")
    features = [f for f in fields if f != target and f not in ID_COLUMNS and f not in LEAKAGE_COLUMNS]
    X, y = [], []
    for number, row in enumerate(rows, 2):
        if any(not str(row.get(f, "")).strip() for f in fields):
            raise DatasetError("blank value at CSV row %d" % number)
        item = {}
        for f in features:
            value = row[f].strip()
            try:
                item[f] = float(value)
            except ValueError:
                item[f] = value
        try:
            effort = float(row[target])
        except ValueError as exc:
            raise DatasetError("non-numeric target at CSV row %d" % number) from exc
        if not math.isfinite(effort):
            raise DatasetError("non-finite target at CSV row %d" % number)
        X.append(item)
        y.append(effort)
    return features, X, y


class EffortGuesser:
    def __init__(self, random_state=20261029, n_estimators=250):
        self.random_state, self.n_estimators = random_state, n_estimators
        self.model = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None

    def fit(self, feature_names, X, y):
        self.feature_names = list(feature_names)
        self.numeric_features = [f for f in self.feature_names if all(isinstance(r[f], (int, float)) for r in X)]
        self.categorical_features = [f for f in self.feature_names if f not in self.numeric_features]
        if not self.feature_names:
            raise DatasetError("no usable features")
        numeric = [self.feature_names.index(f) for f in self.numeric_features]
        categorical = [self.feature_names.index(f) for f in self.categorical_features]
        prep = ColumnTransformer([
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        forest = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state,
                                       min_samples_leaf=2, n_jobs=1)
        self.model = Pipeline([("preprocess", prep), ("regressor", forest)])
        self.model.fit(self._matrix(X), y)
        return self

    def _matrix(self, rows):
        return [[row[f] for f in self.feature_names] for row in rows]

    def predict(self, project):
        if self.model is None:
            raise RuntimeError("fit must be called before predict")
        missing = set(self.feature_names) - set(project)
        extra = set(project) - set(self.feature_names)
        if missing or extra:
            raise ValueError("project keys mismatch; missing=%s extra=%s" % (sorted(missing), sorted(extra)))
        row = {f: project[f] for f in self.feature_names}
        transformed = self.model.named_steps["preprocess"].transform(self._matrix([row]))
        trees = self.model.named_steps["regressor"].estimators_
        values = [tree.predict(transformed)[0] for tree in trees]
        mean = sum(values) / len(values)
        return {"prediction": mean, "uncertainty_std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))}


def evaluate(csv_path, test_size=0.2, random_state=20261029):
    names, X, y = load_dataset(csv_path)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = EffortGuesser(random_state=random_state).fit(names, train_x, train_y)
    predictions = model.model.predict(model._matrix(test_x))
    return {"rows": len(X), "features": len(names), "train_rows": len(train_x), "test_rows": len(test_x),
            "mae": mean_absolute_error(test_y, predictions),
            "rmse": math.sqrt(mean_squared_error(test_y, predictions)),
            "r2": r2_score(test_y, predictions)}, model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate NASA93DEM project effort")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--config", nargs="*", help="feature=value pairs")
    args = parser.parse_args(argv)
    metrics, model = evaluate(args.csv)
    print(json.dumps(metrics, sort_keys=True))
    if args.config:
        print(json.dumps(model.predict(dict(item.split("=", 1) for item in args.config)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
