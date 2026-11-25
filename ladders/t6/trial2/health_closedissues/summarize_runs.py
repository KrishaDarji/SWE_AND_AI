#!/usr/bin/env python3
"""Summarize model-configuration quality measurements."""
import csv, math, statistics, sys
from collections import Counter, defaultdict

CONFIG = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOMES = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}
MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "missing"}

def finite_number(value):
    if value is None or value.strip().lower() in MISSING: return None
    try:
        x = float(value.strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def median(values):
    return statistics.median(values) if values else None

def dispersion(values):
    return {"sample_stddev": statistics.stdev(values) if len(values) >= 2 else None,
            "sample_variance": statistics.variance(values) if len(values) >= 2 else None}

def fmt(x):
    return "missing" if x is None else (str(int(x)) if isinstance(x, float) and x.is_integer() else str(x))

def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); rows = list(reader)
    if not rows: return {"path": path, "empty": True, "records": 0, "columns": [], "roles": {}, "malformed": [], "missing": {}, "invalid": {}, "outcomes": {}, "numeric": {}, "criterion": {}, "configurations": {"distinct": 0, "repeated": [], "exact_duplicate_rows": 0}}
    header = [x.strip() for x in rows[0]]
    roles = {c: ("outcome" if c in OUTCOMES else "configuration" if c in CONFIG else "other") for c in header}
    missing = Counter(); invalid = Counter(); records=[]; malformed=[]; exact=Counter()
    for pos, raw in enumerate(rows[1:], 2):
        if len(raw) != len(header): malformed.append(pos); continue
        row = {header[i]: raw[i].strip() for i in range(len(header))}; records.append((pos,row)); exact[tuple(raw)] += 1
        for c in header:
            if row[c].lower() in MISSING: missing[c] += 1
            elif c in NUMERIC or c in OUTCOMES:
                if finite_number(row[c]) is None: invalid[c] += 1
    numeric={}; criterion=Counter(); normalized=[]
    for c in CONFIG:
        vals=[]
        for _,r in records:
            if c not in r: continue
            if c == "criterion":
                if r[c].lower() not in MISSING: criterion[r[c].strip()] += 1
            else:
                x=finite_number(r[c]); vals += [] if x is None else [x]
        if c != "criterion": numeric[c] = stats(vals)
    for _,r in records:
        key=[]; valid=True
        for c in CONFIG:
            if c not in r: valid=False; break
            if c == "criterion": v=r[c].strip(); valid &= v.lower() not in MISSING; key.append(("cat",v))
            else:
                v=finite_number(r[c]); valid &= v is not None; key.append(("num",v))
        if valid: normalized.append(tuple(key))
    counts=Counter(normalized)
    return {"path":path,"empty":False,"records":len(records),"columns":header,"roles":roles,"malformed":malformed,"missing":dict(missing),"invalid":dict(invalid),"outcomes":{c:stats([finite_number(r[c]) for _,r in records if c in r and finite_number(r[c]) is not None]) | {"direction":d} for c,d in OUTCOMES.items()},"numeric":numeric,"criterion":dict(criterion),"configurations":{"distinct":len(counts),"repeated":[{"configuration":list(k),"count":n} for k,n in counts.items() if n>1],"exact_duplicate_rows":sum(n-1 for n in exact.values() if n>1)}}

def stats(vals):
    return {"finite_count":len(vals),"minimum":min(vals) if vals else None,"maximum":max(vals) if vals else None,"mean":statistics.mean(vals) if vals else None,"median":median(vals),**dispersion(vals)}

def render(a):
    if a.get("empty"): return "Empty input: zero records and no columns."
    lines=["Model-configuration quality summary",f"Records: {a['records']}",f"Columns: {', '.join(a['columns'])}",f"Roles: {a['roles']}",f"Malformed data rows (original CSV positions): {a['malformed'] or 'none'}",f"Missing tokens: {sorted(MISSING)}",f"Missing values: {a['missing'] or 'none'}",f"Invalid/nonfinite numeric values: {a['invalid'] or 'none'}","Configuration numeric summaries:"]
    for c,s in a["numeric"].items(): lines.append(f"  {c}: {s}")
    lines.append(f"criterion frequencies (categorical): {a['criterion']}")
    for c,s in a["outcomes"].items(): lines.append(f"{c} ({s['direction']}): {s}")
    lines += [f"Normalized distinct complete configurations: {a['configurations']['distinct']}",f"Repeated normalized configurations: {a['configurations']['repeated']}",f"Exact duplicate data rows beyond first: {a['configurations']['exact_duplicate_rows']}","Normalization: numeric equivalents use float values; categorical whitespace is stripped; missing/invalid settings are excluded from configuration keys.","Dispersion: sample standard deviation and sample variance; both are missing when fewer than two finite observations exist.","Rows retain original CSV positions (header is position 1); malformed rows are excluded. Missing outcomes remain excluded from that outcome's finite summary.","These are model-configuration quality measurements, not an issue-closure timeline."]
    return "\n".join(lines)

def main(argv):
    if len(argv) != 2: print("usage: summarize_runs.py DATASET.csv", file=sys.stderr); return 2
    print(render(analyze(argv[1]))); return 0
if __name__ == "__main__": raise SystemExit(main(sys.argv))
