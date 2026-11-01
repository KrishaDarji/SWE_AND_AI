#!/usr/bin/env python3
"""Robust marginal screening for unusual model-quality outcomes."""
import csv, json, math, statistics, sys
from summarize_runs import CONFIG, OUTCOMES, finite_number

THRESHOLD = 3.5
MIN_SAMPLE = 3
MAD_FACTOR = 1.4826

def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f); raw = list(reader)
    if not raw: return [], [], []
    header = [x.strip() for x in raw[0]]; rows=[]; malformed=[]
    for position, values in enumerate(raw[1:], 2):
        if len(values) != len(header): malformed.append(position); continue
        rows.append((position, {header[i]: values[i].strip() for i in range(len(header))}))
    return header, rows, malformed

def _score(value, center, scale):
    return abs(value-center)/scale if scale is not None and scale > 0 else None

def screen(path, threshold=THRESHOLD, min_sample=MIN_SAMPLE):
    if not math.isfinite(threshold) or threshold < 0: raise ValueError("threshold must be finite and nonnegative")
    header, rows, malformed = _rows(path); outcomes={}; flagged=[]
    for outcome, direction in OUTCOMES.items():
        eligible=[(p,r,finite_number(r.get(outcome, ""))) for p,r in rows]
        eligible=[x for x in eligible if x[2] is not None]
        values=[x[2] for x in eligible]; center=statistics.median(values) if values else None
        mad=statistics.median([abs(x-center) for x in values]) if values else None
        scale=MAD_FACTOR*mad if mad is not None else None
        defined=scale is not None and scale > 0 and len(values) >= min_sample
        statuses={"invalid_or_missing":0,"undefined_score":0,"nonflag":0,"flag":0}
        local=[]
        for p,r in rows:
            value=finite_number(r.get(outcome, ""))
            if value is None: statuses["invalid_or_missing"] += 1; continue
            score=_score(value, center, scale)
            if not defined:
                statuses["undefined_score"] += 1; continue
            if score >= threshold:
                statuses["flag"] += 1
                item={"position":p,"outcome":outcome,"configuration":{k:r.get(k,"") for k in CONFIG},"observed":value,"score":score,"reference":{"count":len(values),"center":center,"mad":mad,"scale":scale,"threshold":threshold,"includes_self":True},"reason":"absolute median/MAD score is at or above threshold"}
                local.append(item); flagged.append(item)
            else: statuses["nonflag"] += 1
        outcomes[outcome]={"direction":direction,"reference_count":len(values),"center":center,"mad":mad,"scale":scale,"threshold":threshold,"minimum_sample":min_sample,"defined":defined,"zero_scale_policy":"MAD=0 leaves scores undefined; no SD substitution","statuses":statuses,"flags":local}
    return {"method":"two-sided robust univariate median/MAD screening; score=abs(value-median)/(1.4826*MAD)","threshold":threshold,"minimum_sample":min_sample,"self_in_reference":True,"malformed_positions":malformed,"position_convention":"header is 1; data rows use original CSV positions; malformed rows are excluded but positions are retained","outcomes":outcomes,"flags":flagged,"unique_flagged_rows":len({x["position"] for x in flagged})}

def render(result):
    lines=["Unusual marginal model-quality observations",result["method"],"Scores are two-sided: unusually good outcomes can flag too; this warrants investigation and does not prove measurement error.","This is marginal screening, not configuration-conditioned anomaly detection or a validated error classifier."]
    for name,x in result["outcomes"].items(): lines.append(f"{name}: reference={x['reference_count']}, center={x['center']}, scale={x['scale']}, threshold={x['threshold']}, defined={x['defined']}, statuses={x['statuses']}, flags={len(x['flags'])}")
    lines.append(f"Unique flagged rows: {result['unique_flagged_rows']}")
    for x in result["flags"][:10]: lines.append(json.dumps(x, allow_nan=False, sort_keys=True))
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: spot.py DATASET.csv")
    print(render(screen(sys.argv[1])))
