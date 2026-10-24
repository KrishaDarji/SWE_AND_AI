#!/usr/bin/env python3
"""Detect suspicious benchmark measurements in a CSV file (standard library only)."""
import argparse
import csv
import math
import statistics


def _median_abs(values, center):
    return statistics.median(abs(x - center) for x in values)


def detect(path, target="Performance-", threshold=3.5):
    """Return a report containing robust global outliers and malformed rows."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or target not in rows[0]:
        raise ValueError("CSV must contain at least one row and the target column")
    values = []
    invalid = []
    for number, row in enumerate(rows, 2):
        try:
            value = float(row[target].strip())
            if not math.isfinite(value):
                raise ValueError
            values.append((number, value))
        except (AttributeError, TypeError, ValueError):
            invalid.append(number)
    if not values:
        raise ValueError("target column has no finite numeric measurements")
    numbers = [value for _, value in values]
    center = statistics.median(numbers)
    mad = _median_abs(numbers, center)
    # MAD=0 is common in small test files; IQR is a conservative fallback.
    if mad:
        scale = mad / 0.6744897501960817
    else:
        q1, _, q3 = statistics.quantiles(numbers, n=4, method="inclusive") if len(numbers) >= 2 else (center, center, center)
        scale = (q3 - q1) / 1.349 if q3 > q1 else 0.0
    outliers = []
    for row_number, value in values:
        score = abs(value - center) / scale if scale else 0.0
        if score > threshold:
            outliers.append({"row": row_number, "value": value, "score": score})
    return {"rows": len(rows), "valid_measurements": len(values), "median": center,
            "mad": mad, "invalid_rows": invalid, "outliers": outliers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--target", default="Performance-")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()
    report = detect(args.csv_path, args.target, args.threshold)
    print("rows={rows} valid={valid_measurements} median={median:.6g} mad={mad:.6g}".format(**report))
    print("invalid_rows=" + (",".join(map(str, report["invalid_rows"])) or "none"))
    if report["outliers"]:
        for item in report["outliers"]:
            print("OUTLIER row={row} value={value:.6g} robust_score={score:.3f}".format(**item))
    else:
        print("outliers=none")


if __name__ == "__main__":
    main()
