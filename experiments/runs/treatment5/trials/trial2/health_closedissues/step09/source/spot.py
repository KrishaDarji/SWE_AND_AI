#!/usr/bin/env python3
"""Marginal robust screen for unusual outcome values."""
import csv
import json
import math
import sys

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
THRESHOLD = 3.5
MAD_FACTOR = 1.4826

def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return None
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2.0

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

def reference(values):
    med = median(values)
    if med is None:
        return {"median": None, "mad": None, "scale": None}
    mad = median([abs(x - med) for x in values])
    return {"median": med, "mad": mad, "scale": MAD_FACTOR * mad}

def assess(value, refs):
    if not finite(value):
        return None, False, "undefined: observed value is missing or nonfinite"
    med, scale = refs["median"], refs["scale"]
    if med is None:
        return None, False, "undefined: no finite reference observations"
    deviation = abs(value - med)
    if scale is None:
        return None, False, "undefined: reference robust scale is unavailable"
    if scale == 0:
        if deviation == 0:
            return 0.0, False, "defined: zero robust scale and value equals reference median"
        return None, False, "undefined: nonzero deviation with zero robust scale"
    score = deviation / scale
    return score, score >= THRESHOLD, ("flagged: absolute robust score meets threshold"
                                       if score >= THRESHOLD else "defined: score below threshold")

def screen(path):
    records = []
    with open(path, newline="", encoding="utf-8") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 1):
            record = {"record_id": str(row_number), "values": {}}
            for name in OUTCOMES:
                raw = row.get(name, "")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
                record["values"][name] = value
            records.append(record)
    outcomes = []
    flagged_ids = set()
    for name in OUTCOMES:
        outcome_records = []
        for record in records:
            peers = [r["values"][name] for r in records
                     if finite(r["values"][name])]
            refs = reference(peers)
            score, flagged, reason = assess(record["values"][name], refs)
            item = {"record_id": record["record_id"], "outcome": name,
                    "observed": record["values"][name], "reference_median": refs["median"],
                    "reference_mad": refs["mad"], "reference_scale": refs["scale"],
                    "score": score, "threshold": THRESHOLD, "flagged": flagged,
                    "reason": reason, "reference_excludes_assessed_record": False}
            outcome_records.append(item)
            if flagged:
                flagged_ids.add(record["record_id"])
        outcomes.append({"outcome": name, "records": outcome_records,
                         "flag_count": sum(1 for item in outcome_records if item["flagged"])})
    return {"method": "median/MAD modified robust score",
            "threshold": THRESHOLD, "outcomes": outcomes,
            "unique_flagged_record_count": len(flagged_ids),
            "flagged_record_ids": sorted(flagged_ids, key=int),
            "interpretation": "Flags are investigation candidates, not confirmed errors. This marginal screen does not condition on configuration settings and may flag legitimate configuration differences; no detection accuracy is claimed without verified labels."}

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: spot.py DATASET.csv")
    result = screen(argv[1])
    print(json.dumps(result, allow_nan=False, sort_keys=True, indent=2))

if __name__ == "__main__":
    main(sys.argv)
