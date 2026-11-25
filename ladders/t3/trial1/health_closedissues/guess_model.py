#!/usr/bin/env python3
"""Predict model error and accuracy from a model configuration table."""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGETS = ["MRE-", "ACC+", "PRED40+"]
FEATURES = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def read_data(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    missing = set(TARGETS + FEATURES) - set(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    x = [[row[k] for k in FEATURES] for row in rows]
    y = [[float(row[k]) for k in TARGETS] for row in rows]
    return x, y


def build_model():
    numeric = [0, 2, 3, 4]
    categorical = [1]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("prep", prep), ("regressor", RandomForestRegressor(
        n_estimators=120, min_samples_leaf=2, random_state=17, n_jobs=1))])


def run(path, test_size=0.25, seed=17):
    x, y = read_data(path)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=seed)
    model = build_model()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    per_target = {}
    for i, target in enumerate(TARGETS):
        actual = [row[i] for row in test_y]
        guess = [row[i] for row in predicted]
        errors = [abs(a - p) for a, p in zip(actual, guess)]
        per_target[target] = {
            "mae": mean_absolute_error(actual, guess),
            "rmse": math.sqrt(mean_squared_error(actual, guess)),
            "r2": r2_score(actual, guess),
            "mean_absolute_error": sum(errors) / len(errors),
        }
    model.fit(x, y)
    return {"rows": len(x), "holdout_rows": len(test_x), "targets": per_target}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.dataset), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
