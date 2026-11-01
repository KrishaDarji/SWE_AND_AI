#!/usr/bin/env python3
"""Describe historical model configurations and their measured outcomes."""
import csv, math, statistics

CONFIGS = {"N_estimators": "numeric", "criterion": "categorical",
           "Min_sample_leaves": "numeric", "Min_impurity_decrease": "numeric",
           "Max_depth": "numeric"}
OUTCOMES = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
MISSING = {"", "?", "NA", "N/A", "null", "None", "nan", "NaN"}

def _number(value):
    if value is None or value.strip() in MISSING: return None
    try:
        x = float(value.strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def _median(xs):
    return statistics.median(xs) if xs else None

def _summary(xs):
    if not xs: return {"finite_count": 0, "minimum": None, "maximum": None,
                       "mean": None, "median": None, "sample_sd": None,
                       "iqr": None, "convention": "insufficient sample: null"}
    sd = statistics.stdev(xs) if len(xs) > 1 else None
    quartiles = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else []
    q1, q3 = (quartiles[0], quartiles[2]) if quartiles else (None, None)
    return {"finite_count": len(xs), "minimum": min(xs), "maximum": max(xs),
            "mean": statistics.fmean(xs), "median": _median(xs), "sample_sd": sd,
            "iqr": None if q1 is None else q3-q1,
            "convention": "sample standard deviation (n-1); inclusive quartiles; null when n<2"}

def analyze(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh); raw_header = next(reader, None)
        if raw_header is None: return empty_result("empty input")
        header = [x.strip() for x in raw_header]
        rows, malformed = [], []
        for pos, cells in enumerate(reader, 2):
            if not cells or all(not x.strip() for x in cells): continue
            if len(cells) != len(header): malformed.append(pos); continue
            rows.append((pos, dict(zip(header, cells))))
    result = empty_result("header-only input" if not rows else None)
    result.update({"record_count": len(rows), "column_names": header,
                   "malformed_row_positions": malformed,
                   "roles": {**{k: "configuration ("+v+")" for k,v in CONFIGS.items()},
                             **{k: "outcome ("+v+")" for k,v in OUTCOMES.items()}}})
    result["missing_values"] = {k: sum(v.get(k, "").strip() in MISSING for _,v in rows) for k in header}
    result["invalid_nonfinite_numeric"] = {k: sum(v.get(k, "").strip() not in MISSING and _number(v.get(k)) is None for _,v in rows)
                                           for k in list(CONFIGS)+list(OUTCOMES) if k in header}
    result["criterion_frequencies"] = {}
    for _, row in rows:
        c = row.get("criterion", "").strip()
        if c not in MISSING: result["criterion_frequencies"][c] = result["criterion_frequencies"].get(c, 0)+1
    result["numeric_settings"] = {k: _summary([_number(r.get(k)) for _,r in rows if _number(r.get(k)) is not None])
                                   for k,v in CONFIGS.items() if v == "numeric" and k in header}
    result["outcomes"] = {k: {"direction": d, **_summary([_number(r.get(k)) for _,r in rows if _number(r.get(k)) is not None])}
                          for k,d in OUTCOMES.items() if k in header}
    normalized, exact = {}, {}
    for pos, row in rows:
        vals = []
        for k, typ in CONFIGS.items():
            val = row.get(k, "").strip()
            if typ == "categorical": val = None if val in MISSING else val
            else:
                n = _number(val); val = None if n is None else n
            vals.append(val)
        key = tuple(vals); normalized.setdefault(key, []).append(pos)
        exact_key = tuple(row.get(k, "") for k in header); exact.setdefault(exact_key, []).append(pos)
    result["normalized_configurations"] = [{"settings": dict(zip(CONFIGS, key)), "row_positions": ps}
                                           for key, ps in normalized.items()]
    result["repeated_configurations"] = [x for x in result["normalized_configurations"] if len(x["row_positions"]) > 1]
    result["exact_duplicate_row_groups"] = [ps for ps in exact.values() if len(ps) > 1]
    result["normalization"] = "Trim categorical whitespace; numeric equivalents use float; missing/invalid settings become null; rows remain position-addressable."
    return result

def empty_result(reason):
    return {"record_count": 0, "column_names": [], "roles": {}, "missing_values": {},
            "invalid_nonfinite_numeric": {}, "criterion_frequencies": {}, "numeric_settings": {},
            "outcomes": {}, "normalized_configurations": [], "repeated_configurations": [],
            "exact_duplicate_row_groups": [], "malformed_row_positions": [], "note": reason}
