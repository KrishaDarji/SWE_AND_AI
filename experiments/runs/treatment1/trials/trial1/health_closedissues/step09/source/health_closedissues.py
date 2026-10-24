#!/usr/bin/env python3
"""Detect suspicious quality measurements in a historical HPO CSV."""
import argparse
import csv
import math
import statistics
import json


def _finite(value):
    try:
        number = float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parameters(values):
    center = statistics.median(values)
    mad = statistics.median(abs(item - center) for item in values)
    if mad:
        scale = mad / 0.6744897501960817
    elif len(values) >= 2:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        scale = (quartiles[2] - quartiles[0]) / 1.349
    else:
        scale = 0.0
    return center, mad, scale


def detect(path, threshold=3.5, measurement_columns=None):
    """Return malformed rows and robust outliers for outcome-like columns."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("CSV has no data rows or header")
    names = list(rows[0])
    if measurement_columns is None:
        # Outcome columns in this dataset use metric suffixes; exclude tunables.
        measurement_columns = [name for name in names if name.endswith(("-", "+"))]
    missing = [name for name in measurement_columns if name not in names]
    if missing:
        raise ValueError("unknown measurement column(s): " + ", ".join(missing))
    report = {"rows": len(rows), "columns": {}}
    for column in measurement_columns:
        values = []
        invalid = []
        for row_number, row in enumerate(rows, 2):
            number = _finite(row.get(column))
            if number is None:
                invalid.append(row_number)
            else:
                values.append((row_number, number))
        if not values:
            report["columns"][column] = {"valid": 0, "invalid_rows": invalid, "outliers": [], "median": None, "scale": 0.0}
            continue
        numbers = [number for _, number in values]
        center, mad, scale = _parameters(numbers)
        outliers = []
        for row_number, number in values:
            score = abs(number - center) / scale if scale else 0.0
            if score > threshold:
                outliers.append({"row": row_number, "value": number, "score": score})
        report["columns"][column] = {"valid": len(values), "invalid_rows": invalid,
                                      "median": center, "mad": mad, "scale": scale, "outliers": outliers}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()
    report = detect(args.csv_path, args.threshold)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
