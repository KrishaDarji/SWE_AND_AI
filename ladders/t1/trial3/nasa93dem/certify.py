#!/usr/bin/env python3
"""Certify whether one NASA93 development process costs less than another."""
import argparse
import csv
import random
import statistics


def load_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def values(rows, process_level):
    return [float(row["Effort-"].strip()) for row in rows
            if row["pmat"].strip() == process_level]


def bootstrap_median_difference(a, b, iterations=5000, seed=20261026):
    """Return percentile CI for median(a)-median(b), using reproducible resampling."""
    rng = random.Random(seed)
    differences = []
    for _ in range(iterations):
        sa = [a[rng.randrange(len(a))] for _ in a]
        sb = [b[rng.randrange(len(b))] for _ in b]
        differences.append(statistics.median(sa) - statistics.median(sb))
    differences.sort()
    low = differences[int(0.025 * (iterations - 1))]
    high = differences[int(0.975 * (iterations - 1))]
    return low, high


def certify(path, cheaper="n", comparator="l"):
    rows = load_rows(path)
    a, b = values(rows, cheaper), values(rows, comparator)
    if not a or not b:
        raise ValueError("both process levels must have observations")
    median_a, median_b = statistics.median(a), statistics.median(b)
    difference = median_a - median_b
    low, high = bootstrap_median_difference(a, b)
    certified = difference < 0 and high < 0
    return {
        "cheaper_process": f"pmat={cheaper}",
        "comparison_process": f"pmat={comparator}",
        "n_cheaper": len(a), "n_comparison": len(b),
        "median_cheaper": median_a, "median_comparison": median_b,
        "median_difference": difference,
        "bootstrap_95_low": low, "bootstrap_95_high": high,
        "certified": certified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--cheaper", default="n")
    parser.add_argument("--comparator", default="l")
    args = parser.parse_args()
    result = certify(args.dataset, args.cheaper, args.comparator)
    print("Genuine lower-cost process: " + (result["cheaper_process"] if result["certified"] else "NOT CERTIFIED"))
    for key in ("n_cheaper", "n_comparison", "median_cheaper", "median_comparison",
                "median_difference", "bootstrap_95_low", "bootstrap_95_high"):
        print(f"{key}: {result[key]}")
    print("certified: " + str(result["certified"]).lower())


if __name__ == "__main__":
    main()
