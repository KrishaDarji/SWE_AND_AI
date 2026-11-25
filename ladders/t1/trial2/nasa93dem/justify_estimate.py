#!/usr/bin/env python3
"""Justify a software estimate from historical NASA project observations."""
import argparse
import csv
import math
import statistics
import sys


def _fit(rows, x_name, y_name):
    points = [(math.log(float(r[x_name])), math.log(float(r[y_name]))) for r in rows]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / denom
    intercept = ybar - slope * xbar
    residuals = [y - (intercept + slope * x) for x, y in points]
    return intercept, slope, residuals


def _prediction(rows, size, y_name):
    intercept, slope, residuals = _fit(rows, "Kloc+", y_name)
    log_size = math.log(size)
    point = math.exp(intercept + slope * log_size)
    spread = statistics.pstdev(residuals)
    # A residual-based band is intentionally simple and explainable: 80% uses
    # approximately +/- 1.28 standard deviations on the log scale.
    factor = math.exp(1.2816 * spread)
    return point, point / factor, point * factor


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Kloc+", "Effort-", "MonthsX"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV must contain Kloc+, Effort-, and MonthsX columns")
    return [r for r in rows if all(float(r[k]) > 0 for k in required)]


def justify(path, size):
    rows = load_rows(path)
    effort, effort_low, effort_high = _prediction(rows, size, "Effort-")
    months, months_low, months_high = _prediction(rows, size, "MonthsX")
    return {
        "size_kloc": size,
        "historical_projects": len(rows),
        "effort_person_months": effort,
        "effort_80_percent_low": effort_low,
        "effort_80_percent_high": effort_high,
        "schedule_months": months,
        "schedule_80_percent_low": months_low,
        "schedule_80_percent_high": months_high,
        "justification": (
            f"For a {size:g} KLOC system, I recommend planning for "
            f"{effort:.1f} person-months and {months:.1f} months. "
            f"This is grounded in {len(rows)} historical NASA projects and "
            "log-linear relationships between size and each outcome. "
            f"An 80% historical prediction band is {effort_low:.1f}-{effort_high:.1f} "
            f"person-months and {months_low:.1f}-{months_high:.1f} months. "
            "The range is the honest protection against project-specific uncertainty; "
            "it should be narrowed after requirements, staffing, and technology are known."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Justify a sponsor-facing project estimate")
    parser.add_argument("dataset")
    parser.add_argument("--kloc", type=float, required=True, help="planned delivered size in KLOC")
    args = parser.parse_args(argv)
    if args.kloc <= 0:
        parser.error("--kloc must be positive")
    result = justify(args.dataset, args.kloc)
    print(result["justification"])
    print(f"Point estimate: {result['effort_person_months']:.1f} person-months; "
          f"{result['schedule_months']:.1f} months")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
