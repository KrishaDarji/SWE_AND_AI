"""Predict model-quality targets from HPO configuration features."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGETS = ("MRE-", "ACC+", "PRED40+")
FEATURES = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
NUMERIC_FEATURES = ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
CATEGORICAL_FEATURES = ("criterion",)


@dataclass
class GuessModel:
    pipeline: Pipeline
    feature_names: Tuple[str, ...]
    target_names: Tuple[str, ...]

    def predict(self, rows: Sequence[Dict[str, object]]) -> List[Dict[str, float]]:
        values = self.pipeline.predict([[_coerce_feature(row, name) for name in self.feature_names] for row in rows])
        return [{name: float(pred[i]) for i, name in enumerate(self.target_names)} for pred in values]


def _coerce_feature(row: Dict[str, object], name: str) -> object:
    if name not in row or row[name] in (None, ""):
        raise ValueError(f"missing feature: {name}")
    return str(row[name]) if name in CATEGORICAL_FEATURES else float(row[name])


def load_dataset(path: str) -> Tuple[List[Dict[str, object]], List[Dict[str, float]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("dataset has no header")
        required = set(FEATURES) | set(TARGETS)
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        features, targets = [], []
        for line, row in enumerate(reader, 2):
            try:
                features.append({name: _coerce_feature(row, name) for name in FEATURES})
                target = {name: float(row[name]) for name in TARGETS}
                if not all(math.isfinite(value) for value in target.values()):
                    raise ValueError("non-finite target")
                targets.append(target)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid row {line}: {exc}") from exc
    if len(features) < 4:
        raise ValueError("need at least four valid rows")
    return features, targets


def build_model(random_state: int = 42) -> GuessModel:
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), [FEATURES.index(name) for name in NUMERIC_FEATURES]),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), [FEATURES.index(name) for name in CATEGORICAL_FEATURES]),
    ])
    estimator = ExtraTreesRegressor(n_estimators=200, random_state=random_state, min_samples_leaf=2, n_jobs=1)
    return GuessModel(Pipeline([("features", transformer), ("model", estimator)]), FEATURES, TARGETS)


def evaluate(path: str, test_size: float = 0.2, random_state: int = 42) -> Dict[str, object]:
    features, targets = load_dataset(path)
    x = [[row[name] for name in FEATURES] for row in features]
    y = [[row[name] for name in TARGETS] for row in targets]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    model = build_model(random_state)
    model.pipeline.fit(x_train, y_train)
    predicted = model.pipeline.predict(x_test)
    metrics = {}
    for i, name in enumerate(TARGETS):
        actual = [row[i] for row in y_test]
        pred = [row[i] for row in predicted]
        metrics[name] = {"mae": mean_absolute_error(actual, pred), "rmse": mean_squared_error(actual, pred) ** 0.5, "r2": r2_score(actual, pred), "within_0.10": sum(abs(a-p) <= 0.10 for a, p in zip(actual, pred)) / len(actual)}
    return {"rows": len(features), "train_rows": len(x_train), "test_rows": len(x_test), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        raise SystemExit("--test-size must be between 0 and 1")
    print(json.dumps(evaluate(args.dataset, args.test_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
