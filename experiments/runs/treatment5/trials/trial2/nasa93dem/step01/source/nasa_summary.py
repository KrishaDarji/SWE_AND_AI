#!/usr/bin/env python3
"""Summarize NASA93 project descriptors, metadata, and delivery outcomes."""
import csv
import math
import sys
from collections import Counter


OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}


def _cell(text):
    value = text.strip()
    if value == "" or value == "?":
        return None
    try:
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(number) if number.is_integer() else number
    except ValueError:
        return value


def _stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "sd": None}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1) if len(values) > 1 else 0
    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": mean, "sd": math.sqrt(variance)}


def read_rows(stream):
    reader = csv.reader(stream, skipinitialspace=False)
    try:
        header = next(reader)
    except StopIteration:
        return [], []
    header = [name.strip() for name in header]
    rows = []
    for raw in reader:
        if not raw or all(not item.strip() for item in raw):
            continue
        if len(raw) != len(header):
            raise ValueError("row has %d fields; expected %d" % (len(raw), len(header)))
        rows.append([_cell(item) for item in raw])
    return header, rows


def summarize(stream):
    header, rows = read_rows(stream)
    for outcome in OUTCOMES:
        if outcome in header:
            position = header.index(outcome)
            if any(row[position] is not None and not isinstance(row[position], (int, float))
                   for row in rows):
                raise ValueError("invalid numeric outcome in %s" % outcome)
    columns = []
    for index, name in enumerate(header):
        values = [row[index] for row in rows if row[index] is not None]
        numeric = bool(values) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        columns.append({"name": name, "index": index, "type": "numeric" if numeric else "categorical",
                        "missing": len(rows) - len(values), "values": values,
                        "statistics": _stats(values) if numeric else None,
                        "frequencies": dict(sorted(Counter(str(v) for v in values).items())) if not numeric else None})
    roles = []
    for name in header:
        if name.endswith("X"):
            role = "metadata (excluded from future modeling)"
        elif name in OUTCOMES:
            role = "outcome"
        else:
            role = "project descriptor"
        roles.append({"name": name, "role": role, "optimization": OUTCOMES.get(name)})
    return {"record_count": len(rows), "column_count": len(header), "columns": columns, "roles": roles,
            "outcomes": {name: {"direction": direction, "delivered_size": name == "Kloc+",
                                  "effort": name == "Effort-", "defects": name == "Defects-"}
                         for name, direction in OUTCOMES.items()},
            "assumptions": ["Header names determine roles; X-suffix fields are metadata.",
                            "Blank, '?' and nonfinite values are missing; numeric-looking categories are not inferred from codes.",
                            "Delivered size is reported as an outcome, not automatically used as a predictor."]}


def main(argv=None):
    path = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "-"
    try:
        with (sys.stdin if path == "-" else open(path, newline="", encoding="utf-8-sig")) as stream:
            result = summarize(stream)
        import json
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
