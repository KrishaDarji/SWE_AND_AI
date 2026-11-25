#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings in a CSV."""

import argparse
import csv
import json
import math
import statistics
import sys


def _number(value, row_number, field):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("row {} field {!r} is not numeric: {!r}".format(row_number, field, value))
    if not math.isfinite(result):
        raise ValueError("row {} field {!r} is not finite".format(row_number, field))
    return result


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fields = reader.fieldnames
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError("row {} has more values than the header".format(row_number))
            clean = {}
            for field in fields:
                value = row.get(field, "")
                if value is None or value.strip() == "":
                    raise ValueError("row {} field {!r} is empty".format(row_number, field))
                clean[field] = value.strip()
            rows.append(clean)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return fields, rows


def summarize(path):
    fields, raw_rows = load_rows(path)
    quality_fields = [field for field in fields if field.endswith(("+", "-"))]
    config_fields = [field for field in fields if field not in quality_fields]
    if not quality_fields:
        raise ValueError("no quality fields ending in '+' or '-' were found")
    numeric_rows = []
    for index, row in enumerate(raw_rows, start=2):
        converted = {}
        for field in quality_fields:
            converted[field] = _number(row[field], index, field)
        numeric_rows.append(converted)

    metrics = {}
    for field in quality_fields:
        values = [row[field] for row in numeric_rows]
        metrics[field] = {
            "direction": "maximize" if field.endswith("+") else "minimize",
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    configurations = {}
    for field in config_fields:
        raw_values = [row[field] for row in raw_rows]
        try:
            values = [_number(value, index, field)
                      for index, value in enumerate(raw_values, start=2)]
        except ValueError:
            counts = {}
            for value in raw_values:
                counts[value] = counts.get(value, 0) + 1
            configurations[field] = {"type": "categorical",
                                     "unique_values": len(counts),
                                     "value_counts": dict(sorted(counts.items()))}
        else:
            configurations[field] = {"type": "numeric",
                                     "unique_values": len(set(values)),
                                     "minimum": min(values), "maximum": max(values),
                                     "mean": statistics.fmean(values),
                                     "median": statistics.median(values)}

    best = {}
    for field in quality_fields:
        key = max if field.endswith("+") else min
        selected = key(numeric_rows, key=lambda row: row[field])
        best[field] = {name: (selected[field] if name == field else raw_rows[numeric_rows.index(selected)][name])
                       for name in fields}
    return {"rows": len(raw_rows), "columns": fields, "quality_metrics": metrics,
            "configuration": configurations, "best_observed": best}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=True))
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
