#!/usr/bin/env python3
"""Summarize encoder settings and the measured Performance- outcome."""
import argparse
import csv
import json
import math
import statistics
import sys

MISSING = {"", "?", "NA", "N/A", "null", "None"}


def _number(value):
    text = value.strip()
    if text in MISSING:
        return None, "missing"
    try:
        number = float(text)
    except ValueError:
        return None, "invalid"
    if not math.isfinite(number):
        return None, "invalid"
    return number, None


def summarize(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {"record_count": 0, "headers": [], "outcome": None,
                    "settings": {}, "outcome_summary": None,
                    "missing_values": 0, "invalid_values": 0,
                    "warnings": ["empty input"]}
        header = [cell.strip() for cell in header]
        outcome_names = [name for name in header if name.endswith("-")]
        outcome = outcome_names[0] if outcome_names else None
        settings = {name: {"frequencies": {}, "missing": 0, "invalid": 0}
                    for name in header if name != outcome}
        values = []
        missing = invalid = records = 0
        warnings = []
        for line_number, row in enumerate(reader, 2):
            if not row or all(not cell.strip() for cell in row):
                continue
            records += 1
            if len(row) != len(header):
                warnings.append("row %d has %d fields; expected %d" %
                                (line_number, len(row), len(header)))
            row = (row + [""] * len(header))[:len(header)]
            for name, raw in zip(header, row):
                if name == outcome:
                    number, problem = _number(raw)
                    if problem == "missing":
                        missing += 1
                    elif problem == "invalid":
                        invalid += 1
                    elif number is not None:
                        values.append(number)
                else:
                    text = raw.strip()
                    if text in MISSING:
                        settings[name]["missing"] += 1
                    else:
                        counts = settings[name]["frequencies"]
                        counts[text] = counts.get(text, 0) + 1
        outcome_summary = None
        if outcome is None:
            warnings.append("no header ending in '-' found")
        elif values:
            ordered = sorted(values)
            outcome_summary = {"name": outcome, "direction": "minimize",
                               "count": len(values), "min": min(values),
                               "max": max(values), "mean": statistics.fmean(values),
                               "median": statistics.median(ordered),
                               "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}
        elif records:
            warnings.append("outcome has no valid numeric values")
        return {"record_count": records, "headers": header, "outcome": outcome,
                "settings": settings, "outcome_summary": outcome_summary,
                "missing_values": missing, "invalid_values": invalid,
                "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.csv_path)
    except (OSError, csv.Error) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
