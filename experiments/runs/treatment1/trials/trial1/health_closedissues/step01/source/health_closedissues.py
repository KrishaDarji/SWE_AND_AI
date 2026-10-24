#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings from a CSV."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter

METRICS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric value")
    return value


def summarize(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no data rows")
    required = set(METRICS) | set(CONFIG)
    missing = required - set(rows[0])
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))

    parsed = []
    for row in rows:
        item = {name: number(row[name]) for name in METRICS}
        item.update({name: row[name] for name in CONFIG})
        parsed.append(item)

    metric_summary = {}
    for name in METRICS:
        values = [row[name] for row in parsed]
        metric_summary[name] = {
            "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
        }

    best = {}
    for name, direction in METRICS.items():
        key = min if direction == "minimize" else max
        winner = key(parsed, key=lambda row: row[name])
        best[name] = {"value": winner[name], "configuration": {k: winner[k] for k in CONFIG}}

    configurations = {}
    for name in CONFIG:
        values = [row[name] for row in parsed]
        counts = Counter(values)
        if name == "criterion":
            configurations[name] = {"unique": len(counts), "counts": dict(sorted(counts.items()))}
        else:
            numeric = [float(value) for value in values]
            configurations[name] = {
                "unique": len(counts), "min": min(numeric), "max": max(numeric),
                "counts": dict(sorted(counts.items())),
            }
    return {"rows": len(parsed), "metrics": metric_summary, "best": best, "configurations": configurations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
