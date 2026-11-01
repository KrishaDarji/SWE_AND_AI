#!/usr/bin/env python3
"""Summarize model configuration and quality columns."""
import csv, math, sys
from collections import Counter

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = ["MRE-", "ACC+", "PRED40+"]
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
REQUIRED = OUTCOMES + CONFIG
MISSING = {"", "?", "na", "nan", "null", "none"}

def clean(value):
    text = "" if value is None else value.strip()
    return None if text.lower() in MISSING else text

def number(value):
    if value is None: return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def stats(values):
    xs = sorted(values)
    if not xs: return {"count": 0}
    n = len(xs); mean = sum(xs) / n
    q = lambda p: xs[min(n - 1, int(p * n))]
    return {"count": n, "min": xs[0], "max": xs[-1], "mean": mean,
            "median": xs[(n - 1) // 2] if n % 2 else (xs[n//2-1] + xs[n//2]) / 2,
            "q1": q(.25), "q3": q(.75)}

def summarize(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not headers: return {"record_count": 0, "columns": [], "empty": True}
        headers = [h.strip() for h in headers]
        rows = [{k.strip(): clean(v) for k, v in row.items()} for row in reader]
    columns = {}
    for name in headers:
        vals = [r.get(name) for r in rows]
        parsed = [number(v) for v in vals if v is not None]
        valid = [x for x in parsed if x is not None]
        numeric = len(valid) == len(parsed)
        columns[name] = {"role": "configuration" if name in CONFIG else "outcome" if name in OUTCOMES else "other",
                         "type": "numeric" if numeric else "categorical", "missing": sum(v is None for v in vals)}
        if numeric: columns[name]["statistics"] = stats(valid)
        else: columns[name]["frequencies"] = dict(sorted(Counter(v for v in vals if v is not None).items()))
    configurations = [tuple(r.get(k) for k in CONFIG) for r in rows]
    counts = Counter(configurations)
    repeats = {"distinct": len(counts), "repeated_configuration_count": sum(n > 1 for n in counts.values()),
               "repeated_row_count": sum(n for n in counts.values() if n > 1),
               "duplicate_rows": sum(n - 1 for n in counts.values() if n > 1)}
    exact = Counter(tuple(r.get(h) for h in headers) for r in rows)
    repeats["duplicate_rows_exact"] = sum(n - 1 for n in exact.values() if n > 1)
    return {"record_count": len(rows), "columns": headers, "column_details": columns,
            "configuration_fields": CONFIG, "outcome_fields": OUTCOMES, "optimization_directions": DIRECTIONS,
            "configuration_counts": repeats, "empty": len(rows) == 0}

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: analyzer.py DATASET.csv")
    result = summarize(argv[1])
    print("Model-quality summary")
    print("records:", result["record_count"])
    print("columns:", ", ".join(result["columns"]))
    for name, detail in result.get("column_details", {}).items():
        print(name, detail)
    print("configuration_counts:", result.get("configuration_counts", {}))
    print("roles:", result.get("configuration_fields", []), result.get("outcome_fields", []))
    print("directions:", result.get("optimization_directions", {}))

if __name__ == "__main__": main(sys.argv)
