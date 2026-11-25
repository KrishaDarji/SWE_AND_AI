#!/usr/bin/env python3
"""Standard-library statistical certification for the X264 measurements."""
import csv
import math
import random
import sys
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs)


def percentile(xs, p):
    ys = sorted(xs)
    pos = (len(ys) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def analyze(path, seed=20261026, bootstrap=10000):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Performance-" not in rows[0]:
        raise ValueError("missing Performance- column")
    values = [float(row["Performance-"]) for row in rows]
    cut = percentile(values, 0.25)
    improved = [x for x in values if x <= cut]
    comparison = [x for x in values if x > cut]
    observed = mean(comparison) - mean(improved)  # positive means improvement

    rng = random.Random(seed)
    boot = []
    for _ in range(bootstrap):
        a = [improved[rng.randrange(len(improved))] for _ in improved]
        b = [comparison[rng.randrange(len(comparison))] for _ in comparison]
        boot.append(mean(b) - mean(a))
    low, high = percentile(boot, 0.025), percentile(boot, 0.975)

    # Randomization null: labels are exchangeable under no group difference.
    pooled = improved + comparison
    extreme = 0
    for _ in range(bootstrap):
        rng.shuffle(pooled)
        stat = mean(pooled[:len(improved)]) - mean(pooled[len(improved):])
        if stat <= -observed:
            extreme += 1
    p_value = (extreme + 1) / (bootstrap + 1)
    credible = low > 0 and p_value < 0.05
    return {
        "n": len(values), "q1": cut, "improved_n": len(improved),
        "comparison_n": len(comparison), "improved_mean": mean(improved),
        "comparison_mean": mean(comparison), "difference": observed,
        "ci_low": low, "ci_high": high, "p_value": p_value,
        "credible": credible,
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    result = analyze(Path(argv[0] if argv else "moot/optimize/config/X264_AllMeasurements.csv"))
    print("X264 quality certification")
    print("Interpretation: lower Performance- is better; improvement is top quartile versus remaining rows.")
    for key in ("n", "q1", "improved_n", "comparison_n", "improved_mean", "comparison_mean", "difference", "ci_low", "ci_high", "p_value"):
        print(f"{key}={result[key]:.6f}" if isinstance(result[key], float) else f"{key}={result[key]}")
    print("statistically_credible=" + ("YES" if result["credible"] else "NO"))


if __name__ == "__main__":
    main()
