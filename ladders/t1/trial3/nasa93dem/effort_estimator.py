#!/usr/bin/env python3
"""Estimate software effort from the NASA93 COCOMO-style dataset."""

import argparse
import csv
import math
from pathlib import Path


def load_observations(csv_path):
    """Read positive KLOC/effort observations from the named CSV columns."""
    observations = []
    with Path(csv_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                kloc = float(row["Kloc+"])
                effort = float(row["Effort-"])
            except (KeyError, TypeError, ValueError):
                continue
            if kloc > 0 and effort > 0:
                observations.append((kloc, effort))
    if len(observations) < 2:
        raise ValueError("dataset must contain at least two positive observations")
    return observations


def fit_log_linear(observations):
    """Fit log(effort) = intercept + slope * log(KLOC)."""
    xs = [math.log(x) for x, _ in observations]
    ys = [math.log(y) for _, y in observations]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("dataset has no KLOC variation")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    return intercept, slope


def estimate_effort(kloc, observations):
    """Return predicted effort and the fitted model parameters."""
    if kloc <= 0:
        raise ValueError("KLOC must be positive")
    intercept, slope = fit_log_linear(observations)
    return math.exp(intercept + slope * math.log(kloc)), intercept, slope


def main(argv=None):
    parser = argparse.ArgumentParser(description="Estimate effort for a proposed software project")
    parser.add_argument("--dataset", required=True, help="NASA93 CSV path")
    parser.add_argument("--kloc", type=float, required=True, help="proposed size in thousands of lines")
    parser.add_argument("--budget", type=float, help="optional effort budget for approval")
    args = parser.parse_args(argv)
    observations = load_observations(args.dataset)
    estimate, intercept, slope = estimate_effort(args.kloc, observations)
    print(f"observations={len(observations)}")
    print(f"estimated_effort={estimate:.2f} person-months")
    print(f"model=exp({intercept:.6f} + {slope:.6f}*ln(KLOC))")
    if args.budget is not None:
        print(f"decision={'APPROVE' if estimate <= args.budget else 'REJECT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
