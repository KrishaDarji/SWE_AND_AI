#!/usr/bin/env python3
"""Independent certification analysis for the X264 measurements."""
import argparse
import csv
import random
import statistics


def read_values(path):
    with open(path, newline="") as handle:
        rows = csv.DictReader(handle)
        if "Performance-" not in (rows.fieldnames or []):
            raise ValueError("missing Performance- column")
        values = [float(row["Performance-"]) for row in rows]
    if len(values) < 10:
        raise ValueError("at least 10 measurements are required")
    return values


def mean_difference(values):
    ordered = sorted(values)
    n_low = max(1, len(ordered) // 10)
    low = ordered[:n_low]
    rest = ordered[n_low:]
    return statistics.mean(rest) - statistics.mean(low), low, rest


def cohens_d(low, rest):
    pooled = (((len(low) - 1) * statistics.variance(low) +
               (len(rest) - 1) * statistics.variance(rest)) /
              (len(low) + len(rest) - 2)) ** 0.5
    return (statistics.mean(rest) - statistics.mean(low)) / pooled if pooled else 0.0


def bootstrap_ci(low, rest, iterations=10000, seed=20261029):
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        a = [low[rng.randrange(len(low))] for _ in low]
        b = [rest[rng.randrange(len(rest))] for _ in rest]
        estimates.append(statistics.mean(b) - statistics.mean(a))
    estimates.sort()
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]


def permutation_pvalue(low, rest, iterations=10000, seed=20261030):
    rng = random.Random(seed)
    observed = statistics.mean(rest) - statistics.mean(low)
    pooled = list(low) + list(rest)
    count = 1  # add-one correction
    for _ in range(iterations):
        rng.shuffle(pooled)
        trial = statistics.mean(pooled[len(low):]) - statistics.mean(pooled[:len(low)])
        if trial >= observed:
            count += 1
    return count / (iterations + 1)


def analyze(path, bootstrap_iterations=10000, permutation_iterations=10000):
    values = read_values(path)
    difference, low, rest = mean_difference(values)
    ci = bootstrap_ci(low, rest, bootstrap_iterations)
    pvalue = permutation_pvalue(low, rest, permutation_iterations)
    effect = cohens_d(low, rest)
    return {
        "n": len(values), "lowest_decile_n": len(low), "remaining_n": len(rest),
        "low_mean": statistics.mean(low), "remaining_mean": statistics.mean(rest),
        "effect_difference": difference, "cohens_d": effect,
        "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1],
        "permutation_pvalue": pvalue,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--permutation-iterations", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(args.dataset, args.bootstrap_iterations, args.permutation_iterations)
    for key, value in result.items():
        print(f"{key}={value:.10f}" if isinstance(value, float) else f"{key}={value}")
    credible = result["effect_difference"] > 0 and result["bootstrap_ci_low"] > 0 and result["permutation_pvalue"] < 0.05
    print("conclusion=" + ("statistically credible quality improvement" if credible else "not statistically credible"))


if __name__ == "__main__":
    main()
