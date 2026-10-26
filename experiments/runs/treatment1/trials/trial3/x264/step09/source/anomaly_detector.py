#!/usr/bin/env python3
"""Detect unusually large/small benchmark measurements in a CSV file."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _quantile(values, fraction):
    return statistics.quantiles(values, n=4, method="inclusive")[int(fraction * 4) - 1]


def detect(path, threshold=3.5):
    """Return a JSON-serializable report using a robust MAD-based rule."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV contains no data rows")
    fields = list(rows[0])
    candidates = []
    for field in fields:
        values = []
        for row in rows:
            try:
                value = float(row[field].strip())
            except (AttributeError, TypeError, ValueError):
                values = []
                break
            if not math.isfinite(value):
                values = []
                break
            values.append(value)
        if values:
            candidates.append((field, values))
    if not candidates:
        raise ValueError("CSV contains no finite numeric column")
    # Benchmark files normally put the response after configuration columns.
    field, values = candidates[-1]
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad:
        scores = [0.6744897501960817 * (value - median) / mad for value in values]
    else:
        scores = [0.0 if value == median else math.inf for value in values]
    flagged = []
    for index, (row, score, value) in enumerate(zip(rows, scores, values), 1):
        if abs(score) >= threshold:
            flagged.append({"row": index, "value": value, "modified_z": score,
                            "configuration": {k: v for k, v in row.items() if k != field}})
    return {"measurement": field, "row_count": len(rows), "median": median,
            "mad": mad, "threshold": threshold, "anomalies": flagged}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args(argv)
    print(json.dumps(detect(args.csv_path, args.threshold), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
