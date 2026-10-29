#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings from a CSV."""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


QUALITY_COLUMNS = ("MRE-", "ACC+", "PRED40+")
CONFIG_COLUMNS = (
    "N_estimators",
    "criterion",
    "Min_sample_leaves",
    "Min_impurity_decrease",
    "Max_depth",
)


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric_summary(values):
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 0.5),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
    }


def spearman_rank_correlation(first, second):
    """Compute Spearman correlation using average ranks and no dependencies."""
    if len(first) != len(second) or not first:
        return None

    def ranks(values):
        indexed = sorted(enumerate(values), key=lambda pair: pair[1])
        output = [0.0] * len(values)
        index = 0
        while index < len(indexed):
            end = index + 1
            while end < len(indexed) and indexed[end][1] == indexed[index][1]:
                end += 1
            rank = (index + end - 1) / 2.0 + 1.0
            for position in range(index, end):
                output[indexed[position][0]] = rank
            index = end
        return output

    x, y = ranks(first), ranks(second)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denominator_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    return numerator / (denominator_x * denominator_y) if denominator_x and denominator_y else None


def summarize(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = set(QUALITY_COLUMNS + CONFIG_COLUMNS)
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError("missing required columns: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("input CSV contains no data rows")

    quality = {name: [float(row[name]) for row in rows] for name in QUALITY_COLUMNS}
    configuration = {}
    for name in CONFIG_COLUMNS:
        values = [row[name] for row in rows]
        counts = Counter(values)
        configuration[name] = {
            "unique": len(counts),
            "most_common": [{"value": value, "count": count} for value, count in counts.most_common(10)],
        }
    best = {}
    for name, values in quality.items():
        # The signs in the source names define optimization direction: MRE is minimized;
        # ACC and PRED40 are maximized.
        index = min(range(len(values)), key=values.__getitem__) if name == "MRE-" else max(range(len(values)), key=values.__getitem__)
        best[name] = {"value": values[index], "row_number": index + 2, "configuration": {key: rows[index][key] for key in CONFIG_COLUMNS}}

    correlations = {}
    for metric in QUALITY_COLUMNS:
        correlations[metric] = {config: spearman_rank_correlation(quality[metric], [float(row[config]) for row in rows]) for config in ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")}
    return {
        "input": str(path),
        "row_count": len(rows),
        "quality_metrics": {name: numeric_summary(values) for name, values in quality.items()},
        "best_configurations": best,
        "configuration_settings": configuration,
        "spearman_correlations": correlations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON to this path")
    args = parser.parse_args(argv)
    result = json.dumps(summarize(args.csv_path), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
