#!/usr/bin/env python3
"""Recommend an x264 configuration and justify it from measured CSV rows."""

import argparse
import csv
import statistics
import sys


def read_measurements(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset contains no measurements")
    if "Performance-" not in rows[0]:
        raise ValueError("dataset must contain a Performance- column")
    for row in rows:
        row["Performance-"] = float(row["Performance-"])
    return rows


def recommend(rows):
    feature_names = [name for name in rows[0] if name != "Performance-"]
    best_value = min(row["Performance-"] for row in rows)
    best = next(row for row in rows if row["Performance-"] == best_value)
    baseline = statistics.mean(row["Performance-"] for row in rows)
    ties = sum(row["Performance-"] == best_value for row in rows)
    return {
        "features": feature_names,
        "configuration": {name: best[name] for name in feature_names},
        "best_performance": best_value,
        "baseline_performance": baseline,
        "improvement_percent": (baseline - best_value) / baseline * 100,
        "measurement_count": len(rows),
        "best_tie_count": ties,
    }


def format_justification(result):
    settings = ", ".join(
        "%s=%s" % (name, result["configuration"][name])
        for name in result["features"]
    )
    tie_note = " It is tied for the best measured result." if result["best_tie_count"] > 1 else ""
    return (
        "Recommended x264 configuration: %s\n"
        "Evidence: this configuration records Performance-=%.3f, the lowest "
        "value among %d measured configurations (dataset mean %.3f).\n"
        "Relative to the dataset mean, the measured improvement is %.2f%%.%s\n"
        "Administrator action: use this configuration as the initial production "
        "candidate, then validate it on representative workloads and confirm "
        "quality, throughput, and resource constraints before rollout.\n"
        "Limitation: this is a recommendation from observed measurements, not a "
        "causal guarantee; workload drift and feature interactions may change the result."
    ) % (
        settings,
        result["best_performance"],
        result["measurement_count"],
        result["baseline_performance"],
        result["improvement_percent"],
        tie_note,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to X264_AllMeasurements.csv")
    args = parser.parse_args(argv)
    try:
        result = recommend(read_measurements(args.dataset))
    except (OSError, ValueError, KeyError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(format_justification(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
