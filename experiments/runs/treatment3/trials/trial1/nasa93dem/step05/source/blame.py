#!/usr/bin/env python3
"""Identify project factors associated with excessive software effort."""
import argparse
import csv
import json
import math
import os
import sys
from statistics import median

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_dataset(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def prepare(rows):
    target_values = [float(row["Effort-"]) for row in rows]
    cutoff = median(target_values)
    excluded = {"Effort-", "Defects-", "MonthsX", "idX"}
    feature_names = [name for name in rows[0] if name not in excluded]
    categorical, numeric = [], []
    for name in feature_names:
        try:
            [float(row[name].strip()) for row in rows]
            numeric.append(name)
        except ValueError:
            categorical.append(name)
    X = [[row[name].strip() if name in categorical else float(row[name].strip())
          for name in feature_names] for row in rows]
    y = [int(value > cutoff) for value in target_values]
    return X, y, feature_names, categorical, numeric, cutoff, target_values


def analyze(path, seed=17):
    rows = read_dataset(path)
    X, y, names, categorical, numeric, cutoff, efforts = prepare(rows)
    categorical_indices = [names.index(name) for name in categorical]
    numeric_indices = [names.index(name) for name in numeric]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
        ("numeric", StandardScaler(), numeric_indices),
    ])
    model = Pipeline([
        ("features", transformer),
        ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    model.fit(X, y)
    folds = min(5, min(y.count(0), y.count(1)))
    if folds < 2:
        raise ValueError("both target classes need at least two observations")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accuracy = float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean())
    permutation = permutation_importance(model, X, y, n_repeats=20, random_state=seed,
                                         scoring="accuracy", n_jobs=1)
    ranked = sorted(zip(names, permutation.importances_mean, permutation.importances_std),
                    key=lambda item: item[1], reverse=True)
    high = sum(y)
    return {
        "rows": len(rows), "cutoff_effort": cutoff, "high_effort_rows": high,
        "high_effort_rate": high / len(rows), "cross_validated_accuracy": accuracy,
        "factors": [{"factor": n, "importance": float(i), "std": float(s)}
                    for n, i, s in ranked],
        "interpretation": "Factors with positive permutation importance are associated with distinguishing excessive-effort projects; importance is predictive association, not causation."
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
