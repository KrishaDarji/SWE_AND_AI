#!/usr/bin/env python3
"""Summarize encoder settings and measured performance from an x264 CSV."""

import argparse
import csv
import json
import math
from pathlib import Path


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": median,
        "stdev": math.sqrt(variance),
    }


def summarize(path, top=5, measure="Performance-"):
    """Return a JSON-serializable summary of historical encoder runs.

    Lower measured performance values are reported first because the supplied
    x264 benchmark uses a dash in the measure name and appears to be a cost.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if measure not in fieldnames:
        raise ValueError("measure column not found: " + measure)
    if not rows:
        raise ValueError("dataset has no data rows")

    settings = [name for name in fieldnames if name != measure]
    values = [_number(row[measure]) for row in rows]
    if any(value is None for value in values):
        raise ValueError("measure column contains a non-numeric value")
    ranked = sorted(zip(values, rows), key=lambda item: item[0])

    setting_summary = {}
    for name in settings:
        counts = {}
        for row in rows:
            value = row.get(name, "")
            counts[value] = counts.get(value, 0) + 1
        setting_summary[name] = {
            "distinct_values": len(counts),
            "counts": dict(sorted(counts.items())),
        }

    def record(item):
        value, row = item
        return {"measure": value, "settings": {name: row[name] for name in settings}}

    return {
        "dataset": str(path),
        "rows": len(rows),
        "settings": settings,
        "measure": measure,
        "measure_summary": _stats(values),
        "setting_summary": setting_summary,
        "best_runs": [record(item) for item in ranked[:top]],
        "worst_runs": [record(item) for item in reversed(ranked[-top:])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--measure", default="Performance-")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    print(json.dumps(summarize(args.dataset, args.top, args.measure), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
