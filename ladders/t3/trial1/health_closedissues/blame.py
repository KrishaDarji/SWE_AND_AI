#!/usr/bin/env python3
"""Identify hyperparameters associated with high model error.

The input is a CSV containing MRE- and model-configuration columns.  The
analysis is deliberately self-contained: CSV parsing uses the standard
library and the predictive association model uses scikit-learn.
"""
import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


TARGET = "MRE-"


def read_data(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or TARGET not in reader.fieldnames:
            raise ValueError("CSV must contain an MRE- column")
        rows = [row for row in reader if row.get(TARGET, "") != ""]
    if not rows:
        raise ValueError("CSV contains no usable rows")
    features = [name for name in rows[0] if name != TARGET]
    return rows, features


def numeric_values(rows, name):
    values = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (ValueError, TypeError, KeyError):
            return None
    return values


def group_summary(rows, feature, high_cutoff):
    groups = {}
    for row in rows:
        groups.setdefault(row[feature], []).append(float(row[TARGET]))
    overall = statistics.mean(float(row[TARGET]) for row in rows)
    result = []
    for value, errors in groups.items():
        result.append({
            "value": value,
            "count": len(errors),
            "mean_error": statistics.mean(errors),
            "high_error_rate": sum(error >= high_cutoff for error in errors) / len(errors),
            "difference_from_overall": statistics.mean(errors) - overall,
        })
    return sorted(result, key=lambda item: item["mean_error"], reverse=True)


def analyze(path: str, seed: int = 42):
    rows, features = read_data(path)
    y = [float(row[TARGET]) for row in rows]
    high_cutoff = statistics.quantiles(y, n=4, method="inclusive")[2] if len(y) >= 4 else max(y)
    categorical = [name for name in features if numeric_values(rows, name) is None]
    numeric = [name for name in features if name not in categorical]
    X = [[row[name] if name in categorical else float(row[name]) for name in features] for row in rows]
    categorical_indices = [features.index(name) for name in categorical]
    numeric_indices = [features.index(name) for name in numeric]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ("numeric", "passthrough", numeric_indices),
    ])
    model = Pipeline([
        ("encode", transformer),
        ("model", RandomForestRegressor(n_estimators=250, random_state=seed, n_jobs=1, min_samples_leaf=3)),
    ])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    permutation = permutation_importance(model, x_test, y_test, n_repeats=8, random_state=seed, scoring="neg_mean_absolute_error", n_jobs=1)
    importance = {name: {"mean": float(permutation.importances_mean[index]), "std": float(permutation.importances_std[index])} for index, name in enumerate(features)}
    ranked = sorted(features, key=lambda name: importance[name]["mean"], reverse=True)
    return {
        "rows": len(rows),
        "target": TARGET,
        "overall_mean_error": statistics.mean(y),
        "high_error_cutoff": high_cutoff,
        "high_error_definition": "MRE- at or above the inclusive 75th percentile",
        "model": {"test_mae": mean_absolute_error(y_test, predictions), "test_r2": r2_score(y_test, predictions)},
        "permutation_importance": [{"feature": name, **importance[name]} for name in ranked],
        "high_error_groups": {name: group_summary(rows, name, high_cutoff)[:10] for name in features},
        "interpretation": "Higher mean_error and high_error_rate indicate association with high error; association is not causal.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    result = analyze(args.csv_path)
    payload = json.dumps(result, indent=2, sort_keys=False)
    if args.output == "-":
        print(payload)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
