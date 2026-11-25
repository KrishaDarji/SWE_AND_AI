#!/usr/bin/env python3
"""Certify whether the worst outcome decile differs from the remainder."""

import argparse
import csv
import math
import random
from pathlib import Path


def read_performance(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "Performance-" not in (reader.fieldnames or []):
            raise ValueError("dataset must contain Performance-")
        values = []
        for row in reader:
            value = row["Performance-"].strip()
            if value:
                values.append(float(value))
    if not values:
        raise ValueError("dataset contains no Performance- values")
    return values


def mean(values):
    return sum(values) / len(values)


def sample_mean(values, rng):
    return mean([values[rng.randrange(len(values))] for _ in values])


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] + fraction * (sorted_values[high] - sorted_values[low])


def pooled_sd(first, second):
    first_mean, second_mean = mean(first), mean(second)
    ss = sum((x - first_mean) ** 2 for x in first) + sum((x - second_mean) ** 2 for x in second)
    return math.sqrt(ss / (len(first) + len(second) - 2))


def certify(values, bootstrap_reps=10000, permutation_reps=10000, seed=20261029):
    ordered = sorted(values)
    decile_size = max(1, math.ceil(len(ordered) * 0.10))
    remaining = ordered[:-decile_size]
    worst = ordered[-decile_size:]
    observed = mean(worst) - mean(remaining)
    sd = pooled_sd(worst, remaining)
    effect_size = observed / sd if sd else float("inf")

    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_reps):
        bootstrap.append(sample_mean(worst, rng) - sample_mean(remaining, rng))
    bootstrap.sort()
    ci = (percentile(bootstrap, 0.025), percentile(bootstrap, 0.975))

    labels = [1] * len(worst) + [0] * len(remaining)
    pooled = worst + remaining
    exceedances = 0
    for _ in range(permutation_reps):
        rng.shuffle(labels)
        perm_worst = [pooled[i] for i, label in enumerate(labels) if label]
        perm_remaining = [pooled[i] for i, label in enumerate(labels) if not label]
        if mean(perm_worst) - mean(perm_remaining) >= observed:
            exceedances += 1
    p_value = (exceedances + 1) / (permutation_reps + 1)
    credible = ci[0] > 0 and p_value < 0.05
    return {
        "n": len(values),
        "worst_n": len(worst),
        "remaining_n": len(remaining),
        "worst_mean": mean(worst),
        "remaining_mean": mean(remaining),
        "effect_difference": observed,
        "effect_size_cohens_d": effect_size,
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "permutation_p_one_sided": p_value,
        "credible": credible,
        "seed": seed,
        "bootstrap_reps": bootstrap_reps,
        "permutation_reps": permutation_reps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    result = certify(read_performance(args.dataset))
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
