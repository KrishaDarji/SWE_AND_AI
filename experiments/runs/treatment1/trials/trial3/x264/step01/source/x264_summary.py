#!/usr/bin/env python3
"""Summarize binary encoder settings and numeric measures in a CSV."""

import argparse
import csv
import statistics
import sys


def read_csv(path):
    with open(path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV has no data rows")
    return rows


def summarize(rows):
    fields = list(rows[0])
    settings, measures = [], []
    for field in fields:
        values = [row[field].strip() for row in rows]
        numeric = []
        valid = True
        for value in values:
            try:
                numeric.append(float(value))
            except ValueError:
                valid = False
                break
        if valid and numeric and all(value in (0.0, 1.0) for value in numeric):
            settings.append((field, numeric))
        elif valid and numeric:
            measures.append((field, numeric))
    return {"rows": len(rows), "settings": settings, "measures": measures}


def format_summary(summary):
    lines = ["Historical encoder-run summary", "", "Runs: %d" % summary["rows"]]
    lines += ["", "Encoder settings (binary columns):"]
    for name, values in summary["settings"]:
        enabled = sum(values)
        lines.append("- %s: enabled %d/%d (%.2f%%)" %
                     (name, int(enabled), len(values), 100.0 * enabled / len(values)))
    lines += ["", "Quality/performance measures:"]
    for name, values in summary["measures"]:
        lines.append("- %s: min %.3f, max %.3f, mean %.3f, median %.3f" %
                     (name, min(values), max(values), statistics.fmean(values),
                      statistics.median(values)))
    if not summary["measures"]:
        lines.append("- none detected")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        print(format_summary(summarize(read_csv(args.csv_path))))
    except (OSError, ValueError, csv.Error) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
