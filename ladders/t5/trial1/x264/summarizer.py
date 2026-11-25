#!/usr/bin/env python3
"""Summarize encoder settings and measured outcomes from a CSV file."""
import argparse
import csv
import math
import sys
from collections import Counter

MISSING = {"", "?", "NA", "N/A", "null", "None"}


def is_missing(value):
    return value is None or value.strip() in MISSING


def numeric(value):
    if is_missing(value):
        return None
    try:
        number = float(value.strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stats(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"count": count, "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / count, "median": median}


def summarize(stream):
    reader = csv.DictReader(stream)
    headers = reader.fieldnames or []
    if not headers:
        return {"status": "empty_input", "record_count": 0, "settings": {},
                "outcome": None, "errors": ["missing header"]}
    outcome = next((h for h in headers if h.endswith("-")), None)
    settings = [h for h in headers if h != outcome]
    records = list(reader)
    setting_summary = {}
    for name in settings:
        values = [row.get(name, "") for row in records]
        setting_summary[name] = {"frequencies": dict(sorted(Counter(
            "<missing>" if is_missing(v) else v.strip() for v in values).items()))}
    valid = []
    invalid = 0
    missing = 0
    if outcome:
        for row in records:
            raw = row.get(outcome, "")
            if is_missing(raw):
                missing += 1
            elif numeric(raw) is None:
                invalid += 1
            else:
                valid.append(numeric(raw))
    result = {"status": "ok", "record_count": len(records), "settings": setting_summary,
              "outcome": {"name": outcome, "direction": "lower_is_better",
                           "missing_count": missing, "invalid_count": invalid,
                           "statistics": stats(valid)} if outcome else None,
              "errors": []}
    if not records:
        result["status"] = "header_only"
    elif not outcome:
        result["status"] = "no_outcome_column"
        result["errors"].append("no header ending in '-'")
    elif not valid:
        result["status"] = "no_valid_outcomes"
    return result


def display(result):
    print("Historical run summary")
    print(f"status: {result['status']}")
    print(f"record_count: {result['record_count']}")
    print(f"encoder_settings: {len(result['settings'])}")
    for name, detail in result["settings"].items():
        print(f"setting {name}: {detail['frequencies']}")
    outcome = result["outcome"]
    if outcome:
        print(f"outcome: {outcome['name']} (lower is better)")
        print(f"missing_outcomes: {outcome['missing_count']}")
        print(f"invalid_outcomes: {outcome['invalid_count']}")
        print(f"statistics: {outcome['statistics']}")
    for error in result["errors"]:
        print(f"error: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    try:
        with open(args.csv_path, newline="", encoding="utf-8") as stream:
            result = summarize(stream)
    except (OSError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    display(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
