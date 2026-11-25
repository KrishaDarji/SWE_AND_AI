#!/usr/bin/env python3
"""Independent CSV quality/configuration summarizer."""
import csv, json, math, statistics, sys
from collections import Counter

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = ["MRE-", "ACC+", "PRED40+"]
ALL = OUTCOMES + CONFIG
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}

def clean(value):
    return value.strip() if value is not None else ""

def number(value):
    text = clean(value)
    if text in ("", "?"): return None
    try:
        result = float(text)
        return result if math.isfinite(result) else None
    except ValueError:
        return None

def describe(values):
    if not values: return {"count": 0, "mean": None, "min": None, "max": None, "stdev": None, "median": None}
    return {"count": len(values), "mean": statistics.fmean(values), "min": min(values),
            "max": max(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values)}

def summarize(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=False)
        headers = [clean(x) for x in (reader.fieldnames or [])]
        rows = [{clean(k): clean(v) for k, v in raw.items()} for raw in reader]
    missing_columns = [x for x in ALL if x not in headers]
    numeric = OUTCOMES + [x for x in CONFIG if x != "criterion"]
    numeric_stats = {x: describe([number(r.get(x, "")) for r in rows if number(r.get(x, "")) is not None]) for x in numeric}
    categorical = {"criterion": dict(sorted(Counter(r.get("criterion", "") or "<missing>" for r in rows).items()))}
    missing = {x: sum(number(r.get(x, "")) is None if x in numeric else not clean(r.get(x, "")) for r in rows) for x in ALL}
    invalid = {x: sum(clean(r.get(x, "")) not in ("", "?") and number(r.get(x, "")) is None for r in rows) for x in numeric}
    configs = [tuple(r.get(x, "") or "<missing>" for x in CONFIG) for r in rows]
    counts = Counter(configs)
    repeated = {json.dumps(list(k)): v for k, v in sorted(counts.items(), key=lambda item: (item[0], item[1])) if v > 1}
    return {"record_count": len(rows), "columns": headers, "data_types": {x: ("categorical" if x == "criterion" else "numeric") for x in headers},
            "missing_values": missing, "invalid_or_nonfinite_numeric": invalid, "numeric_statistics": numeric_stats,
            "categorical_frequencies": categorical, "configuration_count_distinct": len(counts),
            "repeated_configurations": repeated, "duplicate_rows": sum(v - 1 for v in Counter(tuple(r.get(x, "") for x in headers) for r in rows).values() if v > 1),
            "configuration_fields": CONFIG, "outcome_fields": OUTCOMES, "optimization_directions": DIRECTIONS,
            "missing_columns": missing_columns, "empty_input": len(rows) == 0, "header_only": bool(headers) and len(rows) == 0,
            "conventions": {"whitespace": "trim surrounding whitespace in headers and cells", "missing": "empty string or ?; invalid/nonfinite numeric values are excluded from numeric statistics and counted", "duplicates": "duplicate_rows counts excess full-row occurrences; repeated configurations count excess configuration occurrences by frequency"}}

def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 1: raise SystemExit("usage: analyzer.py DATASET.csv")
    print(json.dumps(summarize(argv[0]), indent=2, sort_keys=True))

if __name__ == "__main__": main()
