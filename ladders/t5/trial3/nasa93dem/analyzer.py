#!/usr/bin/env python3
"""Summarize a NASA93-style CSV without third-party dependencies."""
import csv, math, statistics, sys
from collections import Counter

MISSING = {"", "?", "na", "n/a", "null", "none", "nan", "+nan", "-nan"}

def parse_cell(value):
    value = value.strip()
    if value.lower() in MISSING:
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return value

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or not any(cell.strip() for cell in rows[0]):
        return [], []
    header = [cell.strip() for cell in rows[0]]
    data = []
    for raw in rows[1:]:
        if not any(cell.strip() for cell in raw):
            continue
        if len(raw) != len(header):
            raise ValueError("row has %d fields; expected %d" % (len(raw), len(header)))
        data.append([parse_cell(cell) for cell in raw])
    return header, data

def role(name):
    if name.endswith("X"):
        return "metadata"
    if name == "Kloc+":
        return "outcome"
    if name in ("Effort-", "Defects-"):
        return "outcome"
    return "descriptor"

def direction(name):
    return {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}.get(name)

def summarize_column(name, values):
    present = [v for v in values if v is not None]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)
    result = {"name": name, "role": role(name), "direction": direction(name),
              "inferred_type": "numeric" if numeric else "categorical", "missing": len(values)-len(present)}
    if numeric and present:
        ordered = sorted(present)
        result["statistics"] = {"count": len(present), "min": min(present), "max": max(present),
            "mean": statistics.fmean(present), "median": statistics.median(ordered),
            "stdev": statistics.stdev(present) if len(present) > 1 else 0.0}
    elif not numeric:
        result["frequencies"] = dict(sorted(Counter(str(v) for v in present).items()))
    return result

def analyze(path):
    header, rows = read_csv(path)
    columns = [summarize_column(name, [row[i] for row in rows]) for i, name in enumerate(header)]
    return {"record_count": len(rows), "columns": columns,
            "outcomes": [c["name"] for c in columns if c["role"] == "outcome"],
            "metadata": [c["name"] for c in columns if c["role"] == "metadata"],
            "descriptors": [c["name"] for c in columns if c["role"] == "descriptor"]}

def render(result):
    lines = ["Historical project summary", "records: %d" % result["record_count"],
             "outcomes: " + ", ".join(result["outcomes"]),
             "metadata (excluded from future modeling): " + ", ".join(result["metadata"]),
             "descriptors: " + ", ".join(result["descriptors"])]
    for c in result["columns"]:
        line = "%s [%s, %s, missing=%d]" % (c["name"], c["role"], c["inferred_type"], c["missing"])
        if c["direction"]:
            line += ", direction=" + c["direction"]
        if "statistics" in c:
            line += ", " + ", ".join("%s=%.4g" % (k, v) for k, v in c["statistics"].items())
        else:
            line += ", frequencies=" + repr(c["frequencies"])
        lines.append(line)
    return "\n".join(lines) + "\n"

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: analyzer.py DATASET.csv")
    sys.stdout.write(render(analyze(argv[1])))

if __name__ == "__main__":
    main(sys.argv)
