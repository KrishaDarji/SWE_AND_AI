#!/usr/bin/env python3
"""Certify whether one configuration family reliably outperforms another."""
import argparse
import csv
import math
import random
import statistics
import sys

METRICS = (("MRE-", -1), ("ACC+", 1), ("PRED40+", 1))


def load_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {name for name, _ in METRICS} | {"criterion"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV is empty or lacks required columns")
    for row in rows:
        for name, _ in METRICS:
            row[name] = float(row[name])
    return rows


def normalized_scores(rows):
    """Return a 0..1 score per row; larger is better, equal metric weights."""
    columns = {}
    for name, direction in METRICS:
        values = [row[name] for row in rows]
        lo, hi = min(values), max(values)
        span = hi - lo
        columns[name] = [0.5] * len(rows) if span == 0 else [
            ((value - lo) / span if direction == 1 else (hi - value) / span)
            for value in values
        ]
    return [sum(columns[name][i] for name, _ in METRICS) / len(METRICS)
            for i in range(len(rows))]


def bootstrap_difference(a, b, iterations=4000, seed=20261026):
    """Bootstrap mean(a)-mean(b), resampling independently within families."""
    rng = random.Random(seed)
    observed = statistics.fmean(a) - statistics.fmean(b)
    diffs = []
    for _ in range(iterations):
        diffs.append(statistics.fmean(rng.choices(a, k=len(a))) -
                     statistics.fmean(rng.choices(b, k=len(b))))
    diffs.sort()
    lo = diffs[int(0.025 * (iterations - 1))]
    hi = diffs[int(0.975 * (iterations - 1))]
    probability = sum(value > 0 for value in diffs) / iterations
    return observed, lo, hi, probability


def certify(rows, family="criterion", iterations=4000):
    scores = normalized_scores(rows)
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault(row[family], []).append(score)
    if len(groups) < 2:
        raise ValueError("at least two configuration families are required")
    ordered = sorted(groups)
    results = []
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            difference, lo, hi, probability = bootstrap_difference(
                groups[left], groups[right], iterations)
            results.append({"left": left, "right": right, "difference": difference,
                            "ci_low": lo, "ci_high": hi, "probability": probability,
                            "reliable": lo > 0 or hi < 0})
    best = max(groups, key=lambda key: statistics.fmean(groups[key]))
    return groups, results, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--iterations", type=int, default=4000)
    args = parser.parse_args(argv)
    if args.iterations < 100:
        parser.error("--iterations must be at least 100")
    rows = load_rows(args.csv_path)
    groups, comparisons, best = certify(rows, iterations=args.iterations)
    print("CERTIFY")
    print(f"rows={len(rows)} families={len(groups)} family_field=criterion")
    for name in sorted(groups):
        print(f"family={name} n={len(groups[name])} mean_score={statistics.fmean(groups[name]):.6f}")
    for item in comparisons:
        status = "reliable" if item["reliable"] else "not_reliable"
        print("comparison={left} vs {right} difference={difference:.6f} "
              "ci95=[{ci_low:.6f},{ci_high:.6f}] p_left_better={probability:.4f} "
              "status={status}".format(status=status, **item))
    print(f"best_mean_family={best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
