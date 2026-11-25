#!/usr/bin/env python3
"""Certify whether two observed process groups differ in project effort."""
import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def summarize(values):
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values)}


def bootstrap_difference(a, b, iterations=5000, seed=1729):
    rng = random.Random(seed)
    observed = statistics.mean(a) - statistics.mean(b)
    draws = []
    for _ in range(iterations):
        aa = [a[rng.randrange(len(a))] for _ in a]
        bb = [b[rng.randrange(len(b))] for _ in b]
        draws.append(statistics.mean(aa) - statistics.mean(bb))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    # A two-sided percentile interval excluding zero is the certification rule.
    return observed, lo, hi, (lo > 0 or hi < 0)


def choose_levels(rows, column):
    counts = {}
    for row in rows:
        value = row.get(column, "").strip()
        counts[value] = counts.get(value, 0) + 1
    levels = sorted(((n, value) for value, n in counts.items() if value), reverse=True)
    if len(levels) < 2:
        raise ValueError("process column must contain at least two non-empty levels")
    return levels[0][1], levels[1][1]


def certify(path, process_column="tool", process_a=None, process_b=None, seed=1729):
    rows = load_rows(path)
    process_a, process_b = process_a or choose_levels(rows, process_column)[0], process_b or choose_levels(rows, process_column)[1]
    usable = [r for r in rows if r.get(process_column, "").strip() in (process_a, process_b) and numeric(r.get("Effort-")) is not None]
    if len(usable) < 10:
        raise ValueError("not enough usable observations")
    target = [numeric(r["Effort-"]) for r in usable]
    feature_names = [k for k in usable[0] if k not in ("Effort-", "Defects-", "MonthsX")]
    categorical = [k for k in feature_names if numeric(usable[0].get(k)) is None]
    numerical = [k for k in feature_names if k not in categorical]
    columns = numerical + categorical
    numeric_indices = list(range(len(numerical)))
    categorical_indices = list(range(len(numerical), len(columns)))
    transformer = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_indices),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_indices),
    ])
    model = Pipeline([("features", transformer), ("regressor", Ridge(alpha=1.0))])
    folds = min(5, len(usable))
    matrix = [[numeric(r[k]) if k in numerical else r[k].strip() for k in columns] for r in usable]
    predictions = cross_val_predict(model, matrix, target, cv=KFold(folds, shuffle=True, random_state=seed))
    a = [target[i] for i, r in enumerate(usable) if r[process_column].strip() == process_a]
    b = [target[i] for i, r in enumerate(usable) if r[process_column].strip() == process_b]
    observed, lo, hi, certified = bootstrap_difference(a, b, seed=seed)
    rmse = math.sqrt(statistics.mean((y - p) ** 2 for y, p in zip(target, predictions)))
    lower_cost = process_a if observed < 0 else process_b
    return {"dataset": str(path), "process_column": process_column, "process_a": process_a,
            "process_b": process_b, "cost_metric": "Effort-", "a": summarize(a), "b": summarize(b),
            "mean_difference_a_minus_b": observed, "bootstrap_95_percent_interval": [lo, hi],
            "genuinely_less_costly": certified, "less_costly_process": lower_cost if certified else None,
            "out_of_fold_rmse": rmse, "decision": (f"{lower_cost} is genuinely less costly than the other process" if certified else "insufficient evidence that either process is genuinely less costly")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--process-column", default="tool")
    parser.add_argument("--process-a")
    parser.add_argument("--process-b")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    print(json.dumps(certify(args.dataset, args.process_column, args.process_a, args.process_b, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
