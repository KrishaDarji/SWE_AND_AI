#!/usr/bin/env python3
"""Marginal robust screen for unusual numeric model-quality outcomes."""
import csv
import json
import math
import statistics
import sys

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
THRESHOLD = 3.5
MAD_CONSTANT = 0.6744897501960817


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def robust_reference(values):
    clean = [x for x in values if x is not None]
    if not clean:
        return {"n": 0, "median": None, "mad": None, "scale": None}
    median = statistics.median(clean)
    mad = statistics.median([abs(x - median) for x in clean])
    return {"n": len(clean), "median": median, "mad": mad,
            "scale": mad / MAD_CONSTANT if mad else 0.0}


def score(value, reference):
    if value is None:
        return None, "missing or nonfinite observed value"
    if reference["n"] == 0:
        return None, "no finite reference observations"
    if reference["scale"] == 0:
        if value == reference["median"]:
            return 0.0, "zero robust scale; observed value equals reference median"
        return None, "zero robust scale; non-median score is undefined"
    return MAD_CONSTANT * (value - reference["median"]) / reference["mad"], None


def screen_rows(rows):
    references = {name: robust_reference([finite_number(row.get(name)) for row in rows])
                  for name in OUTCOMES}
    records = []
    for index, row in enumerate(rows, 1):
        record_id = "data_row_%d" % index
        for outcome in OUTCOMES:
            observed = finite_number(row.get(outcome))
            ref = references[outcome]
            value_score, reason = score(observed, ref)
            flagged = value_score is not None and abs(value_score) >= THRESHOLD
            if flagged:
                reason = "absolute robust score meets or exceeds threshold"
            elif reason is None:
                reason = "defined score is below threshold"
            records.append({
                "record_id": record_id, "outcome": outcome,
                "observed": observed, "reference_median": ref["median"],
                "reference_mad": ref["mad"], "robust_scale": ref["scale"],
                "score": value_score, "threshold": THRESHOLD,
                "flagged": flagged, "reason": reason,
                "reference_includes_record": True,
            })
    counts = {outcome: sum(r["flagged"] for r in records if r["outcome"] == outcome)
              for outcome in OUTCOMES}
    unique = len({r["record_id"] for r in records if r["flagged"]})
    return {"method": "median/MAD modified z score", "threshold": THRESHOLD,
            "reference_includes_record": True, "outcomes": list(OUTCOMES),
            "records": records, "unique_flagged_record_count": unique,
            "per_outcome_flag_counts": counts}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: spot_screen.py DATASET.csv")
    result = screen_rows(read_csv(argv[1]))
    json.dump(result, sys.stdout, allow_nan=False, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
