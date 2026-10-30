#!/usr/bin/env python3
"""Independent, outcome-isolated anomaly spotting for x264 measurements."""
import csv, json, math, sys

THRESHOLD = 3.0
MIN_COMPARISONS = 2

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def mean_sd(values):
    n = 0; mu = 0.0; m2 = 0.0
    for value in values:
        if not finite(value): continue
        n += 1; delta = value - mu; mu += delta / n; m2 += delta * (value - mu)
    return n, mu, math.sqrt(max(m2, 0.0) / (n - 1)) if n > 1 else 0.0

def detect(rows, outcome="Performance-", threshold=THRESHOLD, min_comparisons=MIN_COMPARISONS):
    if not rows: return {"status":"empty_input", "threshold":threshold, "flags":[], "flagged_record_count":0}
    keys = [k for k in rows[0] if k != outcome]
    groups = {}
    for index, row in enumerate(rows, 1):
        key = tuple(row.get(k) for k in keys); groups.setdefault(key, []).append((index, row))
    flags = []; skipped = 0
    for index, row in enumerate(rows, 1):
        value = row.get(outcome)
        if not finite(value): skipped += 1; continue
        key = tuple(row.get(k) for k in keys)
        # Each distinct encoder-setting group contributes one mean. The entire
        # assessed group is excluded, so no assessed outcome can leak into refs.
        reference_groups = []
        for other_key, members in groups.items():
            if other_key == key: continue
            values = [r.get(outcome) for _, r in members if finite(r.get(outcome))]
            if values:
                _, group_mean, _ = mean_sd(values)
                reference_groups.append((other_key, group_mean))
        n = len(reference_groups)
        reference_values = [value for _, value in reference_groups]
        _, reference, sd = mean_sd(reference_values)
        if n < min_comparisons:
            skipped += 1; continue
        if sd == 0.0:
            score = None
            reason = "zero_reference_variability"
            is_flag = False
        else:
            score = abs(value - reference) / sd
            reason = "absolute_studentized_deviation"
            is_flag = score >= threshold
        if is_flag:
            flags.append({"record_id": index, "observed": value, "expected_reference": reference,
                          "score": score, "threshold": threshold, "comparison_count": n,
                          "comparison_group": {k: row.get(k) for k in keys}, "reference_group_ids": [list(k) for k, _ in reference_groups], "reason": reason})
    return {"status":"ok", "outcome":outcome, "direction":"lower_is_better", "threshold":threshold,
            "minimum_comparisons":min_comparisons, "record_identifier":"1-based CSV data-row number",
            "comparison_definition":"mean outcome per distinct encoder-setting group; assessed group excluded entirely; reference variability is sample SD across eligible group means",
            "flags":flags, "flagged_record_count":len(flags), "skipped_record_count":skipped,
            "limitations":["Flags are investigation candidates, not confirmed errors.",
                           "No accuracy claim is made without verified anomaly labels.",
                           "Groups with too few references or zero variability are not scored."]}

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream); rows = []
        for row in reader:
            for key, value in list(row.items()):
                try: row[key] = float(value)
                except (TypeError, ValueError): pass
            rows.append(row)
    return rows

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: anomaly_detector.py DATASET.csv")
    result = detect(read_csv(argv[1]))
    json.dump(result, sys.stdout, allow_nan=False, sort_keys=True); sys.stdout.write("\n")

if __name__ == "__main__": main(sys.argv)
