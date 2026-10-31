#!/usr/bin/env python3
"""Retrospective observed-data hyperparameter repair."""
import csv, math, json, sys

FIELDS = ("N_estimators", "criterion", "Min_sample_leaves",
          "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}

def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)

def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def parse_record(raw, identifier):
    """Return a normalized record, retaining the supplied identifier."""
    r = {"identifier": identifier}
    for key in OUTCOMES:
        r[key] = number(raw.get(key))
    for key in FIELDS:
        if key in NUMERIC:
            x = number(raw.get(key))
            r[key] = int(x) if x is not None and x.is_integer() else x
        else:
            value = raw.get(key)
            r[key] = value.strip() if isinstance(value, str) else value
    return r

def valid_configuration(record):
    return all(record.get(k) is not None for k in FIELDS) and all(
        (finite(record[k]) if k in NUMERIC else bool(record[k])) for k in FIELDS)

def valid_observed_record(record):
    return valid_configuration(record) and finite(record.get("MRE-"))

def same_value(a, b, key):
    if key in NUMERIC:
        return number(a) == number(b)
    return a == b

def changes(baseline, candidate):
    return [{"field": k, "old": baseline[k], "new": candidate[k]}
            for k in FIELDS if not same_value(baseline[k], candidate[k], k)]

def choose(baseline, records):
    if not valid_observed_record(baseline):
        return {"status": "invalid_baseline", "reason": "baseline needs finite MRE- and usable configuration"}
    eligible = []
    for record in records:
        if not valid_observed_record(record):
            continue
        if record["MRE-"] < baseline["MRE-"]:
            diff = changes(baseline, record)
            eligible.append((len(diff), record["MRE-"], -(record["ACC+"] if finite(record["ACC+"]) else -math.inf),
                             -(record["PRED40+"] if finite(record["PRED40+"]) else -math.inf), record["identifier"], record, diff))
    if not eligible:
        return {"status": "no_improvement", "reason": "no eligible observed record has strictly lower finite MRE-"}
    winner = min(eligible, key=lambda item: item[:5])
    _, _, _, _, _, candidate, diff = winner
    result = {"status": "improvement", "baseline_identifier": baseline["identifier"],
              "candidate_identifier": candidate["identifier"],
              "baseline_configuration": {k: baseline[k] for k in FIELDS},
              "candidate_configuration": {k: candidate[k] for k in FIELDS},
              "baseline_outcomes": {k: baseline[k] for k in OUTCOMES},
              "candidate_outcomes": {k: candidate[k] for k in OUTCOMES},
              "changes": diff, "change_count": len(diff),
              "change_count_is_proxy": True,
              "MRE-_improvement": baseline["MRE-"] - candidate["MRE-"]}
    for key in ("ACC+", "PRED40+"):
        old, new = baseline[key], candidate[key]
        result[key + "_difference"] = new - old if finite(old) and finite(new) else None
        result[key + "_worsened"] = (new < old) if finite(old) and finite(new) else None
        if result[key + "_difference"] is None:
            result[key + "_unknown_reason"] = "baseline or candidate secondary outcome is unavailable or nonfinite"
    return result

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return [parse_record(row, index) for index, row in enumerate(csv.DictReader(stream), 1)]

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: repair.py DATASET.csv")
    records = load_csv(argv[1])
    if not records:
        result = {"status": "empty_input", "reason": "CSV has no data records"}
    else:
        result = choose(records[0], records[1:])
    print(json.dumps(result, sort_keys=True, allow_nan=False))

if __name__ == "__main__":
    main(sys.argv)
