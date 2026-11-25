#!/usr/bin/env python3
"""Marginal robust screening of three quality outcomes."""
import csv, json, math, sys
from pathlib import Path

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
THRESHOLD = 3.5
MAD_CONSISTENCY = 0.6744897501960817

def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)

def median(xs):
    ys = sorted(xs)
    if not ys: return None
    n = len(ys); m = n // 2
    return ys[m] if n % 2 else (ys[m-1] + ys[m]) / 2.0

def reference(values):
    good = [v for v in values if finite(v)]
    med = median(good)
    if med is None: return {"n": 0, "median": None, "mad": None, "scale": None, "method": "unavailable"}
    mad = median([abs(v-med) for v in good])
    scale = mad / MAD_CONSISTENCY if mad else 0.0
    method = "median/MAD"
    if scale == 0.0 and len(set(good)) > 1:
        q1 = median(sorted(good)[:len(good)//2])
        q3 = median(sorted(good)[(len(good)+1)//2:])
        scale = (q3-q1) / 1.349 if q3 is not None and q1 is not None else 0.0
        method = "median/IQR fallback"
    return {"n": len(good), "median": med, "mad": mad, "scale": scale, "method": method}

def screen(rows):
    refs = {k: reference([r.get(k) for r in rows]) for k in OUTCOMES}
    flags = []
    counts = {k: 0 for k in OUTCOMES}
    for r in rows:
        for outcome in OUTCOMES:
            value, ref = r.get(outcome), refs[outcome]
            reason = None; score = None; flagged = False
            if not finite(value):
                reason = "undefined: missing or nonfinite observed value"
            elif ref["n"] < 3:
                reason = "undefined: fewer than three finite reference values"
            elif ref["scale"] == 0:
                score = 0.0 if value == ref["median"] else None
                if score == 0.0:
                    reason = "defined below threshold: constant finite reference"
                else:
                    flagged = True
                    reason = "unbounded: zero robust scale with differing value"
            else:
                score = (value-ref["median"]) / ref["scale"]
                flagged = abs(score) > THRESHOLD
                reason = "above threshold: unusually high" if score > THRESHOLD else ("above threshold: unusually low" if score < -THRESHOLD else "defined below threshold")
            if flagged:
                counts[outcome] += 1
                flags.append({"record_id": r["record_id"], "outcome": outcome, "observed": value, "reference_median": ref["median"], "score": score, "threshold": THRESHOLD, "flagged": True, "reason": reason})
    return {"threshold": THRESHOLD, "reference_includes_assessed_record": True, "references": refs, "flags": flags, "unique_flagged_record_count": len({x["record_id"] for x in flags}), "outcome_flag_counts": counts}

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for record_id, row in enumerate(csv.DictReader(f), 1):
            item = {"record_id": record_id}
            for k in OUTCOMES:
                try: item[k] = float(row[k])
                except (KeyError, TypeError, ValueError): item[k] = None
            rows.append(item)
    return rows

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: spot_screen.py DATASET")
    result = screen(load(argv[1]))
    result["limitations"] = ["Marginal screen does not condition on configuration settings.", "Flags are investigation candidates, not confirmed measurement errors.", "No detection accuracy is claimed without verified labels."]
    print(json.dumps(result, allow_nan=False, sort_keys=True))

if __name__ == "__main__": main(sys.argv)
