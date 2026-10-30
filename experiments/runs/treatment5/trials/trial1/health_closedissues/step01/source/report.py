#!/usr/bin/env python3
"""Summarize model-quality metrics and configuration settings."""
import csv, math, sys
from collections import Counter

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
NUMERIC = set(OUTCOMES) | {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}

def parse(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    return fields, rows

def number(value):
    text = "" if value is None else value.strip()
    if text == "" or text == "?": return None
    try:
        x = float(text)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def stats(values):
    values = sorted(values)
    if not values: return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    n = len(values)
    mid = (values[(n - 1)//2] + values[n//2]) / 2
    return {"count": n, "min": values[0], "max": values[-1], "mean": sum(values)/n, "median": mid}

def summarize(path):
    fields, rows = parse(path)
    missing = {k: 0 for k in fields}
    types = {}
    numeric = {}
    categories = {}
    for field in fields:
        vals = [r.get(field, "") for r in rows]
        clean = ["" if v is None else v.strip() for v in vals]
        missing[field] = sum(number(v) is None for v in clean) if field in NUMERIC else sum(v in ("", "?") for v in clean)
        if field in NUMERIC:
            valid = [number(v) for v in clean]
            numeric[field] = stats([v for v in valid if v is not None])
            types[field] = "numeric"
        else:
            categories[field] = dict(sorted(Counter(v for v in clean if v not in ("", "?")).items()))
            types[field] = "categorical"
    config_keys = [tuple((r.get(k, "") or "").strip() for k in CONFIG) for r in rows]
    row_keys = [tuple((r.get(k, "") or "").strip() for k in fields) for r in rows]
    repeated = sum(n for n in Counter(config_keys).values() if n > 1)
    return {"record_count": len(rows), "columns": list(fields), "data_types": types,
            "missing_values": missing, "numeric": numeric, "categorical": categories,
            "roles": {"configuration": list(CONFIG), "outcomes": list(OUTCOMES)},
            "optimization_directions": DIRECTIONS, "distinct_configurations": len(set(config_keys)),
            "repeated_configuration_rows": repeated, "duplicate_rows": len(rows)-len(set(row_keys)),
            "conventions": {"whitespace": "strip each cell", "missing": "blank or ?; numeric nonfinite/invalid also missing",
                            "duplicates": "duplicate_rows counts exact normalized full rows beyond first; repeated_configuration_rows counts all rows in configuration groups of size >1",
                            "empty/header_only": "zero records and zero-valued summaries"}}

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: report.py DATASET.csv")
    print_report(summarize(argv[1]))

def print_report(s):
    print("Model-quality and configuration summary")
    print("record_count:", s["record_count"])
    print("columns:", ", ".join(s["columns"]))
    print("data_types:", s["data_types"])
    print("missing_values:", s["missing_values"])
    for k, v in s["numeric"].items(): print("numeric", k, v)
    for k, v in s["categorical"].items(): print("categorical", k, v)
    print("roles:", s["roles"])
    print("optimization_directions:", s["optimization_directions"])
    print("distinct_configurations:", s["distinct_configurations"])
    print("repeated_configuration_rows:", s["repeated_configuration_rows"])
    print("duplicate_rows:", s["duplicate_rows"])
    print("conventions:", s["conventions"])

if __name__ == "__main__": main(sys.argv)
