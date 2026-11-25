#!/usr/bin/env python3
"""Robust marginal screening for NASA93 project outcomes."""
import csv, json, math, statistics, sys

OUTCOMES = {"Kloc+": "Kloc+", "Effort-": "Effort-", "Defects-": "Defects-"}
THRESHOLD = 3.5

def finite_nonnegative(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None

def median(values):
    return statistics.median(values) if values else None

def reference(values):
    """Return median/MAD on log1p values; the assessed row is included."""
    logs = [math.log1p(x) for x in values]
    center = median(logs)
    mad = median([abs(x - center) for x in logs]) if center is not None else None
    scale = 1.4826 * mad if mad is not None else None
    return center, mad, scale

def score(value, center, mad, scale):
    if value is None:
        return None, "undefined: missing, nonfinite, or negative"
    if center is None or mad is None or scale is None:
        return None, "undefined: empty reference sample"
    if scale == 0:
        if math.log1p(value) == center:
            return 0.0, "constant/zero robust scale; equal to reference center"
        return None, "unbounded: zero robust scale and value differs from center"
    return abs(math.log1p(value) - center) / scale, None

def screen(rows):
    refs = {}
    for field in OUTCOMES.values():
        vals = [r[field] for r in rows if r[field] is not None]
        center, mad, scale = reference(vals)
        refs[field] = {"median_log1p": center, "mad_log1p": mad,
                       "scale_log1p": scale, "valid_n": len(vals)}
    results, flagged = [], set()
    counts = {f: 0 for f in OUTCOMES.values()}
    for row in rows:
        fields = []
        for field in OUTCOMES.values():
            s, reason = score(row[field], refs[field]["median_log1p"],
                              refs[field]["mad_log1p"], refs[field]["scale_log1p"])
            flag = s is not None and s > THRESHOLD
            if flag:
                fields.append({"field": field, "observed": row[field],
                               "reference": refs[field], "score": s,
                               "threshold": THRESHOLD,
                               "reason": "robust log1p MAD score exceeds threshold"})
                counts[field] += 1
            elif reason:
                fields.append({"field": field, "observed": row[field],
                               "reference": refs[field], "score": s,
                               "threshold": THRESHOLD, "reason": reason})
        if any(f["score"] is not None and f["score"] > THRESHOLD for f in fields):
            flagged.add(row["original_position"])
        results.append({"original_position": row["original_position"],
                        "idX": row.get("idX"), "observed": row["observed"],
                        "fields": fields})
    return {"threshold": THRESHOLD, "reference_includes_assessed_record": True,
            "reference": refs, "records": results,
            "unique_flagged_record_count": len(flagged),
            "per_outcome_flag_counts": counts,
            "limitations": ["Flags are investigation candidates, not confirmed errors.",
                            "Large projects may legitimately have large effort or defect counts.",
                            "This marginal screen is not scope-adjusted and provides no accuracy labels or prediction uncertainty."]}

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, raw in enumerate(csv.DictReader(f), 2):
            position = n - 1
            vals = {field: finite_nonnegative(raw.get(field)) for field in OUTCOMES.values()}
            rows.append({"original_position": position, "idX": raw.get("idX"),
                         "observed": dict(vals), **vals})
    return rows

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: spot.py DATASET")
    print(json.dumps(screen(load(argv[1])), allow_nan=False, sort_keys=True))

if __name__ == "__main__":
    main(sys.argv)
