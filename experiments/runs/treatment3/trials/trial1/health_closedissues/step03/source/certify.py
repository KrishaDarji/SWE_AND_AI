#!/usr/bin/env python3
"""Certify whether one model-configuration family reliably beats another."""
import argparse
import csv
import math
import random
import statistics
import time
from pathlib import Path


REQUIRED = ("MRE-", "ACC+", "PRED40+", "criterion")


def read_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(name not in rows[0] for name in REQUIRED):
        raise ValueError("CSV must contain MRE-, ACC+, PRED40+, and criterion columns")
    return rows


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def score_rows(rows):
    """Return per-row equal-weight score after global min-max normalization."""
    columns = [[float(row["MRE-"]) for row in rows],
               [float(row["ACC+"]) for row in rows],
               [float(row["PRED40+"]) for row in rows]]
    bounds = [(min(values), max(values)) for values in columns]
    scores = []
    for row in rows:
        values = [float(row["MRE-"]), float(row["ACC+"]), float(row["PRED40+"])]
        normalized = []
        for index, value in enumerate(values):
            low, high = bounds[index]
            scaled = 0.5 if high == low else (value - low) / (high - low)
            normalized.append(1.0 - scaled if index == 0 else scaled)
        scores.append(sum(normalized) / len(normalized))
    return scores


def certify(rows, bootstrap=2000, seed=20261029):
    scores = score_rows(rows)
    families = {}
    for row, score in zip(rows, scores):
        families.setdefault(row["criterion"], []).append(score)
    if len(families) < 2:
        raise ValueError("At least two criterion families are required")
    names = sorted(families)
    rng = random.Random(seed)
    means = {name: statistics.fmean(values) for name, values in families.items()}
    results = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            a, b = families[left], families[right]
            observed = means[left] - means[right]
            diffs = []
            for _ in range(bootstrap):
                sample_a = [a[rng.randrange(len(a))] for _ in a]
                sample_b = [b[rng.randrange(len(b))] for _ in b]
                diffs.append(statistics.fmean(sample_a) - statistics.fmean(sample_b))
            low, high = percentile(diffs, 0.025), percentile(diffs, 0.975)
            if low > 0:
                winner = left
                reliable = True
            elif high < 0:
                winner = right
                reliable = True
            else:
                winner = "none"
                reliable = False
            results.append({"family_a": left, "family_b": right,
                            "mean_a": means[left], "mean_b": means[right],
                            "difference_a_minus_b": observed,
                            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                            "winner": winner, "reliably_better": reliable})
    best = max(means, key=means.get)
    return {"families": {name: {"count": len(values), "mean_score": means[name]}
                          for name, values in sorted(families.items())},
            "comparisons": results, "recommended_family": best,
            "score_definition": "equal mean of min-max normalized MRE (reversed), ACC, and PRED40"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    started = time.perf_counter()
    report = certify(read_rows(args.csv_path), args.bootstrap, args.seed)
    report["runtime_seconds"] = time.perf_counter() - started
    print("CERTIFICATION REPORT")
    print(report["score_definition"])
    for name, details in report["families"].items():
        print(f"{name}: n={details['count']} mean_score={details['mean_score']:.6f}")
    for item in report["comparisons"]:
        print(f"{item['family_a']} vs {item['family_b']}: winner={item['winner']} "
              f"CI=[{item['bootstrap_ci_low']:.6f}, {item['bootstrap_ci_high']:.6f}] "
              f"reliably_better={item['reliably_better']}")
    print(f"recommended_family={report['recommended_family']}")
    print(f"runtime_seconds={report['runtime_seconds']:.6f}")


if __name__ == "__main__":
    main()
