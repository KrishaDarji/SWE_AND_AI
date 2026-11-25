#!/usr/bin/env python3
"""Predict model quality for the Health-ClosedIssues HPO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGETS = ["MRE-", "ACC+", "PRED40+"]
DEFAULT_DATASET = Path("moot/optimize/hpo/Health-ClosedIssues0000.csv")


def load_dataset(path: str | Path) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    """Load, validate, and return rows, configuration feature names, and targets."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    with dataset_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("dataset has no header")
        missing = [name for name in TARGETS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing target columns: {missing}")
        features = [name for name in reader.fieldnames if name not in TARGETS]
        if not features:
            raise ValueError("dataset has no configuration features")
        rows = list(reader)
    if len(rows) < 10:
        raise ValueError(f"dataset needs at least 10 rows, found {len(rows)}")
    for index, row in enumerate(rows, start=2):
        for name in TARGETS + features:
            if row.get(name, "").strip() == "":
                raise ValueError(f"blank value in {name} at CSV line {index}")
        for name in TARGETS:
            try:
                value = float(row[name])
            except ValueError as exc:
                raise ValueError(f"non-numeric target {name} at CSV line {index}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite target {name} at CSV line {index}")
    return rows, features, TARGETS.copy()


def _numeric_columns(rows: Sequence[Dict[str, str]], features: Sequence[str]) -> List[str]:
    numeric = []
    for name in features:
        try:
            for row in rows:
                float(row[name])
        except ValueError:
            continue
        numeric.append(name)
    return numeric


def train_guess_model(path: str | Path = DEFAULT_DATASET, *, random_state: int = 3,
                      test_size: float = 0.2) -> Tuple[Pipeline, Dict[str, Any]]:
    """Fit a leakage-safe model and return it with held-out prediction metrics."""
    rows, features, targets = load_dataset(path)
    x = [[row[name] for name in features] for row in rows]
    y = [[float(row[name]) for name in targets] for row in rows]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    numeric = _numeric_columns(rows, features)
    categorical = [name for name in features if name not in numeric]
    numeric_indices = [features.index(name) for name in numeric]
    categorical_indices = [features.index(name) for name in categorical]
    transformer = ColumnTransformer([
        ("numeric", "passthrough", numeric_indices),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
    ])
    model = Pipeline([
        ("preprocess", transformer),
        ("regressor", RandomForestRegressor(
            n_estimators=120, random_state=random_state, n_jobs=-1, min_samples_leaf=2
        )),
    ])
    model.fit(x_train, y_train)
    model.feature_names_ = features
    prediction = model.predict(x_test)
    metrics: Dict[str, Any] = {
        "rows": len(rows), "train_rows": len(x_train), "test_rows": len(x_test),
        "features": features, "targets": targets, "numeric_features": numeric,
        "categorical_features": categorical,
        "mae_overall": float(mean_absolute_error(y_test, prediction)),
        "rmse_overall": float(mean_squared_error(y_test, prediction) ** 0.5),
        "r2_overall": float(r2_score(y_test, prediction, multioutput="uniform_average")),
        "per_target": {},
    }
    for index, target in enumerate(targets):
        actual = [row[index] for row in y_test]
        predicted = [row[index] for row in prediction]
        metrics["per_target"][target] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted)),
        }
    return model, metrics


def predict(model: Pipeline, configurations: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """Predict target values for configuration dictionaries."""
    feature_names = getattr(model, "feature_names_", None)
    if feature_names is None:
        raise ValueError("model does not contain feature names; train it with train_guess_model")
    values = [[configuration[name] for name in feature_names] for configuration in configurations]
    return model.predict(values).tolist()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", action="store_true", help="emit machine-readable metrics")
    args = parser.parse_args(argv)
    try:
        _, metrics = train_guess_model(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print(f"rows={metrics['rows']} train={metrics['train_rows']} test={metrics['test_rows']}")
        print(f"MAE={metrics['mae_overall']:.6f} RMSE={metrics['rmse_overall']:.6f} R2={metrics['r2_overall']:.6f}")
        for target, values in metrics["per_target"].items():
            print(f"{target}: MAE={values['mae']:.6f} RMSE={values['rmse']:.6f} R2={values['r2']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
