#!/usr/bin/env python3
"""Suggest the smallest configuration change likely to improve a model."""

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
    required = set(NUMERIC + CATEGORICAL + TARGETS)
    if not rows:
        raise ValueError("dataset contains no rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    return rows


def _ranges(rows):
    result = {}
    for name in NUMERIC:
        values = [float(row[name]) for row in rows]
        result[name] = (min(values), max(values) - min(values) or 1.0)
    return result


def distance(configuration, row, ranges):
    value = sum(((float(configuration[name]) - float(row[name])) / ranges[name][1]) ** 2
                 for name in NUMERIC)
    value += sum(str(configuration[name]) != str(row[name]) for name in CATEGORICAL)
    return math.sqrt(value)


def predict(rows, configuration, neighbors=5):
    """Inverse-distance prediction of outcomes for a configuration."""
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    for name in NUMERIC + CATEGORICAL:
        if name not in configuration:
            raise ValueError("configuration missing " + name)
    ranges = _ranges(rows)
    ranked = sorted((distance(configuration, row, ranges), i, row)
                    for i, row in enumerate(rows))[:min(neighbors, len(rows))]
    if ranked[0][0] == 0:
        return {name: float(ranked[0][2][name]) for name in TARGETS}
    weights = [1.0 / item[0] for item in ranked]
    total = sum(weights)
    return {name: sum(w * float(item[2][name]) for w, item in zip(weights, ranked)) / total
            for name in TARGETS}


def suggest(rows, baseline, neighbors=5):
    """Return the nearest observed configuration with a predicted improvement."""
    base = predict(rows, baseline, neighbors)
    ranges = _ranges(rows)
    candidates = []
    for index, row in enumerate(rows):
        candidate = {name: row[name] for name in NUMERIC + CATEGORICAL}
        if all(str(candidate[name]) == str(baseline[name]) for name in candidate):
            continue
        # Each candidate is an observed configuration, so its row is the
        # measured prediction; this avoids an O(rows^2) nested neighbor search.
        prediction = {name: float(row[name]) for name in TARGETS}
        improves = prediction["MRE-"] < base["MRE-"] or prediction["ACC+"] > base["ACC+"]
        if not improves:
            continue
        change = distance(baseline, candidate, ranges)
        gain = max(base["MRE-"] - prediction["MRE-"], prediction["ACC+"] - base["ACC+"])
        candidates.append((change, -gain, index, candidate, prediction))
    if not candidates:
        return dict(baseline), base, False
    change, _, _, candidate, prediction = min(candidates)
    return candidate, prediction, True


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
    baseline = {"N_estimators": args.n_estimators, "criterion": args.criterion,
                "Min_sample_leaves": args.min_sample_leaves,
                "Min_impurity_decrease": args.min_impurity_decrease,
                "Max_depth": args.max_depth}
    rows = read_rows(args.dataset)
    recommendation, result, improved = suggest(rows, baseline, args.neighbors)
    print("improved=" + str(improved).lower())
    print("configuration=" + ",".join(f"{k}={recommendation[k]}" for k in NUMERIC + CATEGORICAL))
    print("predicted_error_mre=" + f"{result['MRE-']:.6f}")
    print("predicted_accuracy=" + f"{result['ACC+']:.6f}")
    print("predicted_pred40=" + f"{result['PRED40+']:.6f}")


if __name__ == "__main__":
    main()
