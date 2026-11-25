#!/usr/bin/env python3
"""Summarize x264 encoder settings and historical quality measurements."""

import argparse
import csv
import json
import math
from pathlib import Path


def _summary(numbers):
    ordered = sorted(numbers)
    n = len(ordered)
    mean = sum(ordered) / n
    middle = n // 2
    median = ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    variance = sum((x - mean) ** 2 for x in ordered) / n
    return {"count": n, "min": min(ordered), "max": max(ordered),
            "mean": mean, "median": median, "stdev": math.sqrt(variance)}


def summarize(dataset, top=5, measure="Performance-"):
    """Read a CSV and return a JSON-serializable historical-run summary."""
    if top < 1:
        raise ValueError("top must be positive")
    with Path(dataset).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)
    if measure not in columns:
        raise ValueError("measure column not found: " + measure)
    if not rows:
        raise ValueError("dataset has no data rows")
    settings = [column for column in columns if column != measure]
    measurements = []
    for row_number, row in enumerate(rows, 2):
        try:
            measurements.append(float(row[measure].strip()))
        except (AttributeError, TypeError, ValueError):
            raise ValueError("non-numeric measure at CSV row " + str(row_number))
    ranked = sorted(zip(measurements, rows), key=lambda pair: pair[0])

    def run(pair):
        value, row = pair
        return {"measure": value, "settings": {name: row[name] for name in settings}}

    distributions = {}
    for name in settings:
        counts = {}
        for row in rows:
            counts[row[name]] = counts.get(row[name], 0) + 1
        distributions[name] = {"distinct_values": len(counts),
                               "counts": dict(sorted(counts.items()))}
    return {
        "dataset": str(dataset), "rows": len(rows), "settings": settings,
        "measure": measure, "measure_summary": _summary(measurements),
        "setting_summary": distributions,
        "best_runs": [run(pair) for pair in ranked[:top]],
        "worst_runs": [run(pair) for pair in reversed(ranked[-top:])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--measure", default="Performance-")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    print(json.dumps(summarize(args.dataset, args.top, args.measure), indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
