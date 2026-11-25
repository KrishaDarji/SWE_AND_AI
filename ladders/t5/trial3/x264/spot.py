#!/usr/bin/env python3
"""Leave-one-configuration-out robust anomaly spotting for X264 measurements."""
import csv, json, math, sys
from collections import defaultdict

THRESHOLD = 3.5
MIN_GROUPS = 3
EPSILON = 1e-12

def median(values):
    if not values: return None
    x = sorted(values); n = len(x); m = n // 2
    return x[m] if n % 2 else (x[m - 1] + x[m]) / 2.0

def mad(values, center):
    return median([abs(x - center) for x in values]) if values else None

def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def detect(rows, feature_names, outcome_name="Performance-", threshold=THRESHOLD,
           min_groups=MIN_GROUPS):
    """Return strict-JSON-compatible result; each row is assessed independently."""
    groups = defaultdict(list)
    clean = []
    for index, row in enumerate(rows, 1):
        key = tuple(row.get(name) for name in feature_names)
        value = finite_number(row.get(outcome_name))
        item = {"record_id": index, "configuration": list(key), "observed": value}
        clean.append(item)
        if value is not None and all(v in (0, 1, "0", "1") for v in key):
            groups[key].append(value)
    flags = []
    records = []
    for item in clean:
        key = tuple(item["configuration"])
        peers = []
        distances = []
        for other_key, values in groups.items():
            if other_key == key: continue
            distance = sum(a != b for a, b in zip(key, other_key))
            distances.append((distance, other_key, values))
        reason = None; reference = score = None; used = 0; actual_distance = None
        if item["observed"] is None:
            reason = "missing_or_nonfinite_outcome"
        elif not distances:
            reason = "no_independent_reference_groups"
        else:
            actual_distance = min(x[0] for x in distances)
            peers = [x for x in distances if x[0] == actual_distance]
            values = [median(x[2]) for x in peers if x[2]]
            used = len(values)
            if used < min_groups:
                reason = "insufficient_nearest_comparison_groups"
            else:
                reference = median(values)
                deviation = mad(values, reference)
                scale = 1.4826 * deviation if deviation is not None else None
                if scale is None or scale <= EPSILON:
                    if item["observed"] != reference:
                        flags.append(item["record_id"])
                        reason = "zero_reference_variability_divergent_observation_score_unbounded"
                    else:
                        reason = "zero_or_unavailable_reference_variability"
                else:
                    score = abs(item["observed"] - reference) / scale
                    if score >= threshold: flags.append(item["record_id"])
        records.append({"record_id": item["record_id"], "observed": item["observed"],
                        "expected_reference": reference, "score": score,
                        "threshold": threshold, "flagged": item["record_id"] in flags,
                        "reason": reason, "comparison_distance": actual_distance,
                        "comparison_group_count": used})
    return {"threshold": threshold, "minimum_comparison_groups": min_groups,
            "flagged_record_ids": flags, "unique_flagged_record_count": len(set(flags)),
            "records": records,
            "method": "nearest Hamming configuration groups; group medians; leave-one-configuration-out MAD",
            "limitations": ["Flags are investigation candidates, not confirmed errors.",
                            "No accuracy claim is made without verified anomaly labels."]}

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def main(argv):
    if len(argv) != 2: raise SystemExit("usage: spot.py DATASET.csv")
    rows = read_csv(argv[1]); fields = list(rows[0]) if rows else []
    features = [x for x in fields if x != "Performance-"]
    result = detect(rows, features)
    json.dump(result, sys.stdout, allow_nan=False, sort_keys=True, indent=2)
    sys.stdout.write("\n")

if __name__ == "__main__": main(sys.argv)
