#!/usr/bin/env python3
"""Justify a software-effort recommendation from NASA93 historical data."""
import argparse
import csv
import math
import statistics
from pathlib import Path


def load_records(path):
    with open(path, newline="") as stream:
        rows = csv.DictReader(stream)
        records = []
        for row in rows:
            try:
                kloc = float(row["Kloc+"])
                effort = float(row["Effort-"])
            except (KeyError, TypeError, ValueError):
                continue
            if kloc > 0 and effort > 0:
                records.append((kloc, effort))
    if not records:
        raise ValueError("dataset contains no positive Kloc+/Effort- records")
    return records


def _median(values):
    return statistics.median(values)


def _log_slope(records):
    xs = [math.log(x) for x, _ in records]
    ys = [math.log(y) for _, y in records]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def analyze(records, target_kloc):
    ratios = [effort / kloc for kloc, effort in records]
    ratio_estimate = _median(ratios) * target_kloc
    slope = _log_slope(records)
    lx = statistics.mean(math.log(x) for x, _ in records)
    ly = statistics.mean(math.log(y) for _, y in records)
    trend_estimate = math.exp(ly + slope * (math.log(target_kloc) - lx))
    # Median evidence is less sensitive to unusually large projects; trend is
    # a cross-check. Recommend the midpoint and expose both inputs to the PM.
    recommended = (ratio_estimate + trend_estimate) / 2
    deviations = [abs(effort - recommended * kloc / target_kloc) for kloc, effort in records]
    return {
        "records": len(records), "target_kloc": target_kloc,
        "median_effort_per_kloc": _median(ratios), "trend_slope": slope,
        "ratio_estimate": ratio_estimate, "trend_estimate": trend_estimate,
        "recommended_effort": recommended,
        "historical_median_effort": _median(e for _, e in records),
        "historical_min_effort": min(e for _, e in records),
        "historical_max_effort": max(e for _, e in records),
        "median_abs_error": _median(deviations),
    }


def render(result):
    return ("NASA93 historical effort justification\n"
            f"Records used: {result['records']}\n"
            f"Target size: {result['target_kloc']:.2f} KLOC\n"
            f"Median historical effort intensity: {result['median_effort_per_kloc']:.2f} person-months/KLOC\n"
            f"Median-intensity estimate: {result['ratio_estimate']:.1f} person-months\n"
            f"Log-log historical trend estimate: {result['trend_estimate']:.1f} person-months (slope {result['trend_slope']:.2f})\n"
            f"Recommended planning estimate: {result['recommended_effort']:.1f} person-months\n"
            f"Historical effort range: {result['historical_min_effort']:.1f} to {result['historical_max_effort']:.1f} person-months\n"
            f"Median absolute deviation from recommendation: {result['median_abs_error']:.1f} person-months\n"
            "Justification: this recommendation combines a robust historical intensity baseline with the dataset's size trend; it is a planning baseline, not a guarantee."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--kloc", type=float, required=True, help="planned software size in KLOC")
    args = parser.parse_args(argv)
    if args.kloc <= 0:
        parser.error("--kloc must be positive")
    print(render(analyze(load_records(args.dataset), args.kloc)))


if __name__ == "__main__":
    main()
