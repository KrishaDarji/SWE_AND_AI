#!/usr/bin/env python3
"""Predict model error and accuracy from prior model configurations."""

import argparse
import csv
import math
from pathlib import Path


NUMERIC = ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
CATEGORICAL = ("criterion",)
TARGETS = ("MRE-", "ACC+", "PRED40+")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset contains no rows")
    required = set(NUMERIC + CATEGORICAL + TARGETS)
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    return rows


def _numeric_ranges(rows):
    ranges = {}
    for name in NUMERIC:
        values = [float(row[name]) for row in rows]
        low, high = min(values), max(values)
        ranges[name] = (low, high - low or 1.0)
    return ranges


def _distance(query, row, ranges):
    distance = 0.0
    for name in NUMERIC:
        low, span = ranges[name]
        distance += ((float(query[name]) - float(row[name])) / span) ** 2
    for name in CATEGORICAL:
        distance += 1.0 if str(query[name]) != str(row[name]) else 0.0
    return math.sqrt(distance)


def predict(rows, configuration, neighbors=5):
    """Return inverse-distance weighted predictions for the target columns."""
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    for name in NUMERIC + CATEGORICAL:
        if name not in configuration:
            raise ValueError("configuration missing " + name)
    ranges = _numeric_ranges(rows)
    ranked = sorted(((_distance(configuration, row, ranges), index, row) for index, row in enumerate(rows)), key=lambda item: item[0])
    selected = ranked[: min(neighbors, len(ranked))]
    if selected[0][0] == 0.0:
        exact = selected[0][2]
        return {name: float(exact[name]) for name in TARGETS}
    weights = [1.0 / item[0] for item in selected]
    total = sum(weights)
    return {
        name: sum(weight * float(item[2][name]) for weight, item in zip(weights, selected)) / total
        for name in TARGETS
    }


def format_prediction(configuration, result):
    return (
        "configuration=" + ",".join(f"{key}={configuration[key]}" for key in NUMERIC + CATEGORICAL)
        + "\n"
        + "predicted_error_mre=" + f"{result['MRE-']:.6f}"
        + "\npredicted_accuracy=" + f"{result['ACC+']:.6f}"
        + "\npredicted_pred40=" + f"{result['PRED40+']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--n-estimators", type=float, required=True)
    parser.add_argument("--criterion", required=True)
    parser.add_argument("--min-sample-leaves", type=float, required=True)
    parser.add_argument("--min-impurity-decrease", type=float, required=True)
    parser.add_argument("--max-depth", type=float, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    configuration = {
        "N_estimators": args.n_estimators,
        "criterion": args.criterion,
        "Min_sample_leaves": args.min_sample_leaves,
        "Min_impurity_decrease": args.min_impurity_decrease,
        "Max_depth": args.max_depth,
    }
    print(format_prediction(configuration, predict(read_rows(args.dataset), configuration, args.neighbors)))


if __name__ == "__main__":
    main()
