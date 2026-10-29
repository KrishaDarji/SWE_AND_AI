#!/usr/bin/env python3
"""Independent configuration justification for Health-ClosedIssues."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

TARGETS = ["MRE-", "ACC+", "PRED40+"]
FEATURES = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    x = []
    y = []
    for row in rows:
        x.append([float(row["N_estimators"]), row["criterion"], float(row["Min_sample_leaves"]),
                  float(row["Min_impurity_decrease"]), float(row["Max_depth"])])
        y.append([float(row[t]) for t in TARGETS])
    return rows, np.asarray(x, dtype=object), np.asarray(y, dtype=float)


def pipeline(regressor):
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), [1])], remainder="passthrough")
    return Pipeline([("prep", prep), ("model", regressor)])


def scores(model, x, y):
    pred = model.predict(x)
    utility = (1.0 - y[:, 0] + y[:, 1] + y[:, 2]) / 3.0
    pred_u = (1.0 - pred[:, 0] + pred[:, 1] + pred[:, 2]) / 3.0
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred, multioutput="uniform_average")),
        "utility_mae": float(mean_absolute_error(utility, pred_u)),
        "utility_rmse": float(np.sqrt(mean_squared_error(utility, pred_u))),
        "utility_r2": float(r2_score(utility, pred_u)),
        "per_target_mae": [float(mean_absolute_error(y[:, i], pred[:, i])) for i in range(3)],
    }


def run(path):
    started = time.perf_counter()
    rows, x, y = load(path)
    train_x, test_x, train_y, test_y, train_i, test_i = train_test_split(
        x, y, np.arange(len(rows)), test_size=0.20, random_state=31415
    )
    models = {
        "random_forest": pipeline(MultiOutputRegressor(RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2, max_features=0.8, random_state=31415, n_jobs=1
        ))),
        "extra_trees": pipeline(MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=180, min_samples_leaf=2, max_features=0.8, random_state=31415, n_jobs=1
        ))),
        "mean_baseline": pipeline(MultiOutputRegressor(DummyRegressor(strategy="mean"))),
    }
    evaluations = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        evaluations[name] = scores(model, test_x, test_y)

    # Recommendation is selected from training observations to avoid using held-out outcomes.
    train_utility = (1.0 - train_y[:, 0] + train_y[:, 1] + train_y[:, 2]) / 3.0
    best_pos = int(np.argmax(train_utility))
    best_index = int(train_i[best_pos])
    recommendation = {k: rows[best_index][k] for k in FEATURES}
    recommendation.update({k: rows[best_index][k] for k in TARGETS})
    recommendation["observed_utility"] = float(train_utility[best_pos])
    elapsed = time.perf_counter() - started
    return {
        "dataset": str(path), "rows": len(rows), "features": FEATURES, "targets": TARGETS,
        "split": {"train": len(train_x), "test": len(test_x), "seed": 31415},
        "evaluations": evaluations, "selected_model": "random_forest",
        "selected_configuration": recommendation, "runtime_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Justify a Health-ClosedIssues configuration")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.dataset)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
