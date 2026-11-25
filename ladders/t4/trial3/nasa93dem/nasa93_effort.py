#!/usr/bin/env python3
"""Grouped, categorical NASA93 effort estimation built on the Step 1 parser."""
import argparse
import csv
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .nasa93_summary import read_csv
except ImportError:
    from nasa93_summary import read_csv

TARGET = "Effort-"
OUTCOMES = {"Kloc+", "Effort-", "Defects-"}


def predictor_names(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if not c.endswith("X") and c not in OUTCOMES]


def _value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "__MISSING__"
    return str(value)


def build_pipeline(names: Sequence[str]) -> Pipeline:
    prep = ColumnTransformer([("categorical", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), list(range(len(names))))], remainder="drop")
    return Pipeline([("preprocess", prep), ("estimator", Ridge(alpha=1.0))])


def _frame(rows: Iterable[Mapping[str, Any]], names: Sequence[str]) -> List[List[str]]:
    return [[_value(row.get(name)) for name in names] for row in rows]


def _usable(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if isinstance(row.get(TARGET), (int, float)) and math.isfinite(float(row[TARGET]))]


def evaluate(path: str, test_size: float = 0.25, random_state: int = 42) -> Dict[str, Any]:
    data = read_csv(path)
    rows = _usable(data["rows"])
    names = predictor_names(data["columns"])
    if len(rows) < 2 or not names:
        raise ValueError("insufficient usable rows or descriptors")
    groups = ["\x1f".join(_value(row.get(name)) for name in names) for row in rows]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(rows, groups=groups))
    train, test = [rows[i] for i in train_idx], [rows[i] for i in test_idx]
    model = build_pipeline(names)
    model.fit(_frame(train, names), [float(row[TARGET]) for row in train])
    predictions = model.predict(_frame(test, names))
    train_mean = float(np.mean([float(row[TARGET]) for row in train]))
    return {"model": model, "predictor_names": names, "train_count": len(train), "test_count": len(test),
            "train_groups": sorted({groups[i] for i in train_idx}), "test_groups": sorted({groups[i] for i in test_idx}),
            "train_rows": train, "train_configurations": sorted({groups[i] for i in train_idx}),
            "test_configurations": sorted({groups[i] for i in test_idx}),
            "mae": float(mean_absolute_error([float(row[TARGET]) for row in test], predictions)),
            "baseline_mae": float(mean_absolute_error([float(row[TARGET]) for row in test], np.full(len(test), train_mean))),
            "training_mean": train_mean, "predictions": predictions.tolist(), "test_rows": test,
            "estimator_settings": {"type": "Ridge", "alpha": 1.0, "split": "GroupShuffleSplit", "test_size": test_size, "random_state": random_state}}


def predict_proposal(model: Pipeline, proposal: Mapping[str, Any], names: Sequence[str]) -> float:
    return float(model.predict(_frame([proposal], names))[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    result = evaluate(args.csv_path)
    print("Predictors: " + ", ".join(result["predictor_names"]))
    print(f"Train/test: {result['train_count']}/{result['test_count']}")
    print(f"Held-out MAE: {result['mae']:.6g}; training-mean baseline MAE: {result['baseline_mae']:.6g}")
    print("Estimator: " + str(result["estimator_settings"]))


if __name__ == "__main__":
    main()
