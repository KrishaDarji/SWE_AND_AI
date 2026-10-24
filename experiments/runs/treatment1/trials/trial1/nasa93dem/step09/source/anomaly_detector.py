#!/usr/bin/env python3
"""Find suspicious effort measurements in the NASA93 project CSV."""
import argparse
import csv
import math
import statistics


def _robust_scale(values):
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    scale = mad / 0.6744897501960817 if mad else 0.0
    if scale == 0.0 and len(values) >= 2:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        scale = (quartiles[2] - quartiles[0]) / 1.349
    return center, mad, scale


def analyze(path, target="Effort-", threshold=3.5):
    """Return robust outliers and basic data-quality findings for *path*."""
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or target not in rows[0]:
        raise ValueError("CSV must contain rows and the requested target column")

    measurements = []
    invalid_rows = []
    for row_number, row in enumerate(rows, start=2):
        try:
            value = float(row[target].strip())
            if not math.isfinite(value) or value < 0:
                raise ValueError
            measurements.append((row_number, value))
        except (AttributeError, TypeError, ValueError):
            invalid_rows.append(row_number)
    if not measurements:
        raise ValueError("target column has no valid measurements")

    values = [value for _, value in measurements]
    center, mad, scale = _robust_scale(values)
    outliers = []
    for row_number, value in measurements:
        score = abs(value - center) / scale if scale else 0.0
        if score > threshold:
            outliers.append({"row": row_number, "id": rows[row_number - 2].get("idX"),
                             "value": value, "score": score})

    ids = {}
    for row_number, row in enumerate(rows, start=2):
        project_id = row.get("idX", "").strip()
        if project_id:
            ids.setdefault(project_id, []).append(row_number)
    duplicate_ids = {key: numbers for key, numbers in ids.items() if len(numbers) > 1}
    return {"rows": len(rows), "valid_measurements": len(measurements),
            "median": center, "mad": mad, "invalid_rows": invalid_rows,
            "duplicate_ids": duplicate_ids, "outliers": outliers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--target", default="Effort-")
    parser.add_argument("--threshold", type=float, default=3.5)
    args = parser.parse_args()
    report = analyze(args.csv_path, args.target, args.threshold)
    print("rows={rows} valid={valid_measurements} median={median:.6g} mad={mad:.6g}".format(**report))
    print("invalid_rows=" + (",".join(map(str, report["invalid_rows"])) or "none"))
    print("duplicate_ids=" + (",".join(sorted(report["duplicate_ids"])) or "none"))
    for item in report["outliers"]:
        print("OUTLIER row={row} id={id} value={value:.6g} robust_score={score:.3f}".format(**item))
    if not report["outliers"]:
        print("outliers=none")


if __name__ == "__main__":
    main()
