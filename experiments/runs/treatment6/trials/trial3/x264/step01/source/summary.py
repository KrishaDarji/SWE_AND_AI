"""Summarize historical encoder runs using only Python's standard library."""
import argparse
import csv
import json
import math
import statistics
import sys

MISSING = {"", "?", "na", "n/a", "null", "none", "missing"}

def is_missing(value):
    return value.strip().lower() in MISSING

def read_runs(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], [], ["input has no header"]
        names = [name.strip() if name else "" for name in reader.fieldnames]
        if any(not name for name in names):
            raise ValueError("header contains an empty column name")
        rows = []
        for number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError("row %d has more fields than the header" % number)
            rows.append({name: (raw.get(name) or "").strip() for name in names})
        return names, rows, []

def numeric_stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "median": None, "stdev": None}
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}

def summarize(path):
    names, rows, warnings = read_runs(path)
    outcome = next((n for n in names if n.rstrip().lower() == "performance-"), None)
    if outcome is None and names:
        raise ValueError("required measured outcome Performance- is absent")
    settings = [n for n in names if n != outcome]
    frequencies = {n: {} for n in settings}
    invalid = []
    outcome_values = []
    missing_outcomes = 0
    for index, row in enumerate(rows, 2):
        for name in settings:
            value = row[name]
            key = "<missing>" if is_missing(value) else value
            frequencies[name][key] = frequencies[name].get(key, 0) + 1
        value = row.get(outcome, "") if outcome else ""
        if is_missing(value):
            missing_outcomes += 1
            continue
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            outcome_values.append(number)
        except ValueError:
            invalid.append({"row": index, "value": value})
    return {"record_count": len(rows), "columns": names, "settings": settings,
            "outcome": {"name": outcome, "direction": "lower is better",
                        "statistics": numeric_stats(outcome_values),
                        "missing_count": missing_outcomes, "invalid": invalid},
            "setting_frequencies": frequencies, "warnings": warnings,
            "assumptions": ["First row is the CSV header.",
                            "Performance- is the sole measured outcome.",
                            "Blank, ?, NA, and common null words are missing."]}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize X264 historical runs")
    parser.add_argument("dataset", help="CSV dataset path")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(summarize(args.dataset), indent=2, sort_keys=True))
    except (OSError, ValueError, csv.Error) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
