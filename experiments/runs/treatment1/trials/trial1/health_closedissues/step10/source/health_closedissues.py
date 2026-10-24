#!/usr/bin/env python3
"""Compare a nearest-neighbor configuration optimizer with random search."""

import argparse
import csv
import random
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


def numeric_ranges(rows):
    result = {}
    for name in NUMERIC:
        values = [float(row[name]) for row in rows]
        low, high = min(values), max(values)
        result[name] = (low, high - low or 1.0)
    return result


def distance(configuration, row, ranges):
    total = 0.0
    for name in NUMERIC:
        low, span = ranges[name]
        total += ((float(configuration[name]) - float(row[name])) / span) ** 2
    total += sum(str(configuration[name]) != str(row[name]) for name in CATEGORICAL)
    return total ** 0.5


def _predict(training, configuration, neighbors, ranges):
    """Inverse-distance prediction of the three objectives."""
    if not training or neighbors < 1:
        raise ValueError("training must be non-empty and neighbors must be positive")
    ranked = sorted(((distance(configuration, row, ranges), index, row)
                     for index, row in enumerate(training)), key=lambda item: item[0])
    selected = ranked[:min(neighbors, len(ranked))]
    if selected[0][0] == 0:
        return {name: float(selected[0][2][name]) for name in TARGETS}
    weights = [1.0 / item[0] for item in selected]
    return {name: sum(weight * float(item[2][name]) for weight, item in zip(weights, selected)) / sum(weights)
            for name in TARGETS}


def predict(training, configuration, neighbors=5):
    """Inverse-distance prediction of the three objectives."""
    return _predict(training, configuration, neighbors, numeric_ranges(training))


def score(row):
    """Higher is better: accuracy, coverage, and inverse error."""
    return (float(row["ACC+"]) + float(row["PRED40+"]) + (1.0 - float(row["MRE-"]))) / 3.0


def optimize(training, candidates, neighbors=5):
    ranges = numeric_ranges(training)
    return max(candidates, key=lambda row: score(_predict(training, row, neighbors, ranges)))


def random_search(training, candidates, seed=7, budget=10):
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(budget, len(candidates)))
    return max(sample, key=score)


def run(rows, holdout_fraction=0.2, seed=7, budget=10, neighbors=5):
    if len(rows) < 3:
        raise ValueError("at least three rows are required")
    split = max(1, min(len(rows) - 1, int(len(rows) * (1.0 - holdout_fraction))))
    training, candidates = rows[:split], rows[split:]
    surrogate = optimize(training, candidates, neighbors)
    baseline = random_search(training, candidates, seed, budget)
    return {"rows": len(rows), "training": len(training), "candidates": len(candidates),
            "surrogate": surrogate, "baseline": baseline,
            "surrogate_score": score(surrogate), "baseline_score": score(baseline)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    result = run(read_rows(args.dataset), args.holdout_fraction, args.seed, args.budget, args.neighbors)
    print("rows={rows} training={training} candidates={candidates}".format(**result))
    print("surrogate_score={:.6f}".format(result["surrogate_score"]))
    print("baseline_score={:.6f}".format(result["baseline_score"]))
    print("surrogate_configuration=" + ",".join(str(result["surrogate"][k]) for k in NUMERIC + CATEGORICAL))
    print("baseline_configuration=" + ",".join(str(result["baseline"][k]) for k in NUMERIC + CATEGORICAL))


if __name__ == "__main__":
    main()
