#!/usr/bin/env python3
"""Independent, standard-library-only comparison for nasa93dem.csv."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

TARGET = "Effort-"
SIZE = "Kloc+"
IGNORE = {"idX", "Effort-", "Defects-", "MonthsX"}


def load_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("dataset is empty")
    for row in rows:
        row[TARGET] = float(row[TARGET])
        row[SIZE] = float(row[SIZE])
    return rows


def _feature_names(rows):
    return [name for name in rows[0] if name not in IGNORE]


def _distance(left, right, names, scales):
    total = 0.0
    for name in names:
        a, b = left[name], right[name]
        try:
            delta = (float(a) - float(b)) / scales[name]
        except (TypeError, ValueError):
            delta = 0.0 if str(a).strip() == str(b).strip() else 1.0
        total += delta * delta
    return math.sqrt(total)


def _scales(training, names):
    result = {}
    for name in names:
        try:
            values = [float(row[name]) for row in training]
        except (TypeError, ValueError):
            continue
        spread = statistics.pstdev(values)
        result[name] = spread if spread > 0 else 1.0
    return result


def knn_prediction(training, query, k=5):
    names = _feature_names(training)
    scales = _scales(training, names)
    neighbors = sorted((_distance(query, row, names, scales), row[TARGET]) for row in training)
    chosen = neighbors[: min(k, len(neighbors))]
    return statistics.mean(value for _, value in chosen)


def historical_prediction(training, query):
    """Historical baseline: median observed effort per KLOC, scaled by query size."""
    ratios = [row[TARGET] / row[SIZE] for row in training if row[SIZE] > 0]
    if not ratios:
        raise ValueError("training data has no positive KLOC values")
    return statistics.median(ratios) * query[SIZE]


def evaluate(rows, k=5):
    knn_errors, baseline_errors = [], []
    for index, query in enumerate(rows):
        training = rows[:index] + rows[index + 1:]
        actual = query[TARGET]
        knn_errors.append(abs(knn_prediction(training, query, k) - actual))
        baseline_errors.append(abs(historical_prediction(training, query) - actual))
    return {
        "projects": len(rows),
        "metric": "MAE effort",
        "estimator_mae": statistics.mean(knn_errors),
        "historical_baseline_mae": statistics.mean(baseline_errors),
        "estimator_better": statistics.mean(knn_errors) < statistics.mean(baseline_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    print(json.dumps(evaluate(load_rows(args.dataset), args.k), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
