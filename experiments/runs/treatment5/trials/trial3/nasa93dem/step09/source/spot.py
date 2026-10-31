#!/usr/bin/env python3
"""Independent robust marginal screen for NASA93 project outcomes."""
import csv, json, math, sys

OUTCOMES = ("Kloc+", "Effort-", "Defects-")

def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def quantile(xs, q):
    if not xs: return None
    ys = sorted(xs); pos = (len(ys) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)

def reference(values):
    """Return log1p median/IQR; reference includes all valid assessed records."""
    logs = [math.log1p(x) for x in values if x is not None and x >= 0]
    if not logs: return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, med, q3 = quantile(logs, .25), quantile(logs, .5), quantile(logs, .75)
    return {"n": len(logs), "median": med, "q1": q1, "q3": q3, "iqr": q3-q1}

def screen(rows, threshold=3.0):
    refs = {}
    for field in OUTCOMES:
        refs[field] = reference([finite_number(r.get(field)) for r in rows])
    results = []
    for row_number, row in enumerate(rows, 2):
        flags = []
        for field in OUTCOMES:
            raw = row.get(field); value = finite_number(raw); ref = refs[field]
            reason = None; score = None
            if value is None: reason = "missing_or_nonfinite"
            elif value < 0: reason = "negative_value"
            elif ref["n"] == 0: reason = "no_valid_reference_values"
            elif ref["iqr"] == 0:
                if math.log1p(value) != ref["median"]: reason = "outside_constant_reference"
                else: reason = "zero_scale_inlier"
            else:
                score = abs(math.log1p(value) - ref["median"]) / ref["iqr"]
                if score > threshold: reason = "robust_log1p_score_exceeds_threshold"
            if reason and (reason not in ("zero_scale_inlier",)):
                flags.append({"field": field, "observed": value, "score": score, "threshold": threshold, "reason": reason, "reference": ref})
        results.append({"row_number": row_number, "identifier": row.get("idX"), "observed": {f: finite_number(row.get(f)) for f in OUTCOMES}, "flags": flags})
    return {"method": "absolute deviation from log1p median divided by log1p IQR", "threshold": threshold, "reference_includes_assessed_record": True, "references": refs, "records": results, "unique_flagged_record_count": sum(bool(x["flags"]) for x in results), "per_outcome_flag_counts": {f: sum(any(z["field"] == f for z in x["flags"]) for x in results) for f in OUTCOMES}}

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh: return list(csv.DictReader(fh, skipinitialspace=True))

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: spot.py DATASET")
    print(json.dumps(screen(load_csv(argv[1])), allow_nan=False, indent=2, sort_keys=True))

if __name__ == "__main__": main(sys.argv)
