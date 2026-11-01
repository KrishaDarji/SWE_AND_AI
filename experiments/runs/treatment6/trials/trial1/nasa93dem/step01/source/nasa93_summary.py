#!/usr/bin/env python3
"""NASA93 descriptive summary, using only the Python standard library."""
import csv, math, statistics, sys
from collections import Counter

MISSING = {"", "?", "na", "n/a", "null", "none"}
OUTCOMES = {"Kloc+": "maximize", "Effort-": "minimize", "Defects-": "minimize"}

def _number(value):
    text = value.strip()
    if text.lower() in MISSING: return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None

def _kind(values):
    present = [v for v in values if v.strip().lower() not in MISSING]
    return "numeric" if present and all(_number(v) is not None for v in present) else "categorical"

def _stats(values):
    xs = [v for v in values if v is not None]
    if not xs: return {"count": 0, "mean": None, "sd": None, "min": None, "median": None, "max": None}
    return {"count": len(xs), "mean": statistics.fmean(xs),
            "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "median": statistics.median(xs), "max": max(xs)}

def read_dataset(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, skipinitialspace=False))
    if not rows: return [], [], []
    header = [cell.strip() for cell in rows[0]]
    if not any(header): return [], [], header
    records = []
    for raw in rows[1:]:
        if not raw or all(not cell.strip() for cell in raw): continue
        cells = list(raw[:len(header)]) + [""] * max(0, len(header) - len(raw))
        records.append({name: cells[i].strip() for i, name in enumerate(header)})
    return header, records, rows

def summarize(path):
    header, records, _ = read_dataset(path)
    kinds = {name: ("numeric" if name in OUTCOMES else _kind([r.get(name, "") for r in records])) for name in header}
    roles = {}
    for name in header:
        roles[name] = ("metadata" if name.endswith("X") else
                       "outcome" if name in OUTCOMES else "descriptor")
    numeric, categorical, missing, invalid = {}, {}, {}, {}
    for name in header:
        vals = [r.get(name, "") for r in records]
        missing[name] = sum(v.strip().lower() in MISSING for v in vals)
        if kinds[name] == "numeric":
            parsed = [_number(v) for v in vals]
            numeric[name] = _stats(parsed)
            invalid[name] = sum(v.strip().lower() not in MISSING and _number(v) is None for v in vals)
        else:
            categorical[name] = dict(sorted(Counter(v for v in vals if v.strip().lower() not in MISSING).items()))
            invalid[name] = 0
    return {"record_count": len(records), "columns": header, "types": kinds,
            "roles": roles, "missing": missing, "invalid_numeric": invalid,
            "numeric": numeric, "categorical": categorical,
            "outcomes": {k: {"direction": v, "role": "outcome"} for k, v in OUTCOMES.items() if k in header},
            "model_predictors": [n for n in header if roles[n] == "descriptor"]}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1: raise SystemExit("usage: nasa93_summary.py DATASET.csv")
    print(__import__("json").dumps(summarize(argv[0]), indent=2, sort_keys=True))

if __name__ == "__main__": main()
