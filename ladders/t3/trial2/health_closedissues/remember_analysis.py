#!/usr/bin/env python3
"""Independent descriptive analysis for Health-ClosedIssues0000.csv."""

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict


QUALITY_COLUMNS = ("MRE-", "ACC+", "PRED40+")
CONFIG_COLUMNS = ("N_estimators", "criterion", "Min_sample_leaves",
                  "Min_impurity_decrease", "Max_depth")
EXPECTED_COLUMNS = QUALITY_COLUMNS + CONFIG_COLUMNS


def _number(value, row_number, column):
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {column} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {column} is not finite")
    return result


def load_dataset(path):
    """Load and validate the expected rectangular CSV schema."""
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EXPECTED_COLUMNS):
            raise ValueError(f"expected columns {list(EXPECTED_COLUMNS)!r}; got {reader.fieldnames!r}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if any(value is None or value.strip() == "" for value in row.values()):
                raise ValueError(f"row {row_number}: missing value")
            parsed = {column: row[column] for column in EXPECTED_COLUMNS}
            for column in QUALITY_COLUMNS + ("N_estimators", "Min_sample_leaves",
                                             "Min_impurity_decrease", "Max_depth"):
                parsed[column] = _number(parsed[column], row_number, column)
            if parsed["N_estimators"] < 1 or parsed["Min_sample_leaves"] < 1 or parsed["Max_depth"] < 1:
                raise ValueError(f"row {row_number}: integer-like configuration must be positive")
            if any(parsed[column] != int(parsed[column]) for column in
                   ("N_estimators", "Min_sample_leaves", "Max_depth")):
                raise ValueError(f"row {row_number}: integer-like configuration is not integral")
            rows.append(parsed)
    if not rows:
        raise ValueError("dataset contains no data rows")
    return rows


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pearson(x_values, y_values):
    x_mean, y_mean = statistics.fmean(x_values), statistics.fmean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values) *
                             sum((y - y_mean) ** 2 for y in y_values))
    return numerator / denominator if denominator else 0.0


def _summary(values):
    return {"min": min(values), "p25": _percentile(values, .25),
            "median": statistics.median(values), "mean": statistics.fmean(values),
            "p75": _percentile(values, .75), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def analyze(rows):
    result = {"rows": len(rows), "quality": {}, "configuration": {},
              "relationships": {"numeric_pearson": {}, "categorical_groups": {}}}
    for column in QUALITY_COLUMNS:
        result["quality"][column] = _summary([row[column] for row in rows])
    for column in CONFIG_COLUMNS:
        values = [row[column] for row in rows]
        if column == "criterion":
            result["configuration"][column] = {"unique": sorted(set(values)),
                                                 "counts": {value: values.count(value) for value in sorted(set(values))}}
        else:
            result["configuration"][column] = _summary(values)
            for quality in QUALITY_COLUMNS:
                key = f"{column} vs {quality}"
                result["relationships"]["numeric_pearson"][key] = _pearson(values, [r[quality] for r in rows])
    for criterion in sorted({row["criterion"] for row in rows}):
        subset = [row for row in rows if row["criterion"] == criterion]
        result["relationships"]["categorical_groups"][criterion] = {
            "rows": len(subset),
            "quality_means": {quality: statistics.fmean(row[quality] for row in subset)
                              for quality in QUALITY_COLUMNS},
        }
    # MRE- is conventionally minimized; the other two quality measures are maximized.
    result["best_observed"] = {
        quality: min(rows, key=lambda row: row[quality])[quality] if quality == "MRE-"
        else max(rows, key=lambda row: row[quality])[quality]
        for quality in QUALITY_COLUMNS
    }
    return result


def format_report(path, result):
    lines = ["Health-ClosedIssues0000 remember-skill analysis",
             f"Input: {path}", f"Validated rows: {result['rows']}", "",
             "Quality metric summaries (MRE- lower is better; ACC+ and PRED40+ higher is better):"]
    for column, summary in result["quality"].items():
        lines.append(column + ": " + ", ".join(f"{key}={value:.6f}" for key, value in summary.items()))
    lines.append("\nConfiguration summaries:")
    for column, summary in result["configuration"].items():
        lines.append(f"{column}: {summary}")
    lines.append("\nNumeric Pearson relationships (configuration vs quality):")
    for key, value in result["relationships"]["numeric_pearson"].items():
        lines.append(f"{key}: {value:.6f}")
    lines.append("\nCriterion-group quality means:")
    for key, value in result["relationships"]["categorical_groups"].items():
        lines.append(f"{key}: {value}")
    lines.extend(["\nAssumptions and limitations:",
                  "- The first three columns are treated as historical quality targets and the remaining columns as configurations.",
                  "- A trailing '-' means lower-is-better for MRE; '+' means higher-is-better for ACC and PRED40.",
                  "- Pearson association is descriptive, not causal, and does not adjust for interactions or confounding.",
                  "- This is an observational historical HPO table: no train/test split, uncertainty intervals, or causal claims are made.",
                  "- Validation checks schema, nonempty finite values, and integral positive count/depth settings; it cannot detect provenance or semantic errors."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args(argv)
    rows = load_dataset(args.dataset)
    print(format_report(args.dataset, analyze(rows)), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
