#!/usr/bin/env python3
"""CSV quality/configuration summary using only Python's standard library."""
import argparse, csv, json, math, os, sys
from collections import Counter

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}

def clean(s):
    if s is None: return ""
    return s.lstrip("\ufeff").strip()

def number(s):
    t = clean(s)
    if not t or t.lower() in {"?", "na", "nan", "null", "none"}: return None
    try:
        x = float(t)
        return x if math.isfinite(x) else None
    except ValueError: return None

def parse(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows: return [], [], 0
    header = [clean(x) for x in rows[0]]
    data = []
    for raw in rows[1:]:
        if not any(clean(x) for x in raw): continue
        vals = [(clean(raw[i]) if i < len(raw) else "") for i in range(len(header))]
        data.append(dict(zip(header, vals)))
    return header, data, max((len(r) for r in rows[1:]), default=len(header))

def stats(values):
    xs = [x for x in values if x is not None]
    if not xs: return {"count": 0, "missing_or_invalid": len(values)}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return {"count": len(xs), "missing_or_invalid": len(values)-len(xs), "min": min(xs), "max": max(xs), "mean": mean, "stdev": math.sqrt(var)}

def report(path):
    header, rows, widest = parse(path)
    numeric = {k: stats([number(r.get(k, "")) for r in rows]) for k in header if k != "criterion"}
    categorical = {"criterion": dict(sorted(Counter(clean(r.get("criterion", "")) or "<missing>" for r in rows).items()))} if "criterion" in header else {}
    keys = []
    for r in rows:
        keys.append(tuple(r.get(k, "") for k in CONFIG))
    freq = Counter(keys)
    repeated = {"|".join(k): n for k, n in sorted(freq.items()) if n > 1}
    invalid = {k: sum(number(r.get(k, "")) is None for r in rows) for k in header if k != "criterion"}
    roles = {k: ("configuration" if k in CONFIG else "outcome" if k in OUTCOMES else "other") for k in header}
    return {"record_count": len(rows), "columns": header, "widest_data_row": widest, "data_types": {k: ("categorical" if k == "criterion" else "numeric") for k in header}, "missing_or_invalid": invalid, "numeric_statistics": numeric, "categorical_frequencies": categorical, "roles": roles, "optimization_directions": OUTCOMES, "distinct_configurations": len(freq), "repeated_configurations": repeated, "duplicate_rows": sum(n-1 for n in freq.values() if n > 1), "counting_conventions": "Rows are nonblank data records; configuration identity is the exact whitespace-normalized cell tuple in the five configuration columns; duplicate rows are excess occurrences of identical full configuration tuples."}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    args = ap.parse_args()
    print(json.dumps(report(args.csv_path), indent=2, sort_keys=False))

if __name__ == "__main__": main()
