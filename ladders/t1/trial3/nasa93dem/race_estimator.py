#!/usr/bin/env python3
"""Independent NASA93 effort estimator and historical baseline comparison."""
import argparse
import csv
import math
import statistics
import time
from pathlib import Path


ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}
TARGET = "Effort-"
FEATURES = ["prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx", "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon", "apex", "plex", "ltex", "tool", "site", "sced"]


def load_rows(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in FEATURES + [TARGET, "Kloc+"]:
            if key in row:
                row[key] = row[key].strip()
    return rows


def vector(row):
    values = []
    for name in FEATURES:
        raw = row[name].lower()
        values.append(float(ORDINAL.get(raw, 3)))
    return values


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def predict_knn(train, row, k=7):
    neighbors = sorted((distance(vector(row), vector(other)), float(other[TARGET])) for other in train)
    chosen = neighbors[:min(k, len(neighbors))]
    weights = [1.0 / (d + 0.25) for d, _ in chosen]
    return sum(weight * value for weight, (_, value) in zip(weights, chosen)) / sum(weights)


def median_baseline(train, _row):
    return statistics.median(float(row[TARGET]) for row in train)


def mae(actual, predicted):
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def evaluate(rows, test_indices, predictor):
    test_set = set(test_indices)
    train = [row for index, row in enumerate(rows) if index not in test_set]
    test = [rows[index] for index in test_indices]
    actual = [float(row[TARGET]) for row in test]
    predicted = [predictor(train, row) for row in test]
    return mae(actual, predicted)


def run(dataset):
    rows = load_rows(dataset)
    # Fixed, transparent holdout: every third record is test data.
    test_indices = list(range(2, len(rows), 3))
    estimator_mae = evaluate(rows, test_indices, predict_knn)
    baseline_mae = evaluate(rows, test_indices, median_baseline)
    return {"rows": len(rows), "test_rows": len(test_indices), "estimator_mae": estimator_mae, "baseline_mae": baseline_mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(args.dataset)
    result["runtime_seconds"] = time.perf_counter() - started
    print("NASA93 race comparison")
    print("protocol: fixed every-third-record holdout; metric: effort MAE")
    print("rows={rows} test_rows={test_rows}".format(**result))
    print("estimator_mae={estimator_mae:.6f}".format(**result))
    print("historical_baseline_mae={baseline_mae:.6f}".format(**result))
    print("winner={}".format("estimator" if result["estimator_mae"] < result["baseline_mae"] else "historical_baseline"))
    print("runtime_seconds={runtime_seconds:.6f}".format(**result))


if __name__ == "__main__":
    main()
