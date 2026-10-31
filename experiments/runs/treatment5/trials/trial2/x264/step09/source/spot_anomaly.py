"""Outcome-isolated anomaly spotting for X264 benchmark rows."""
import csv, json, math, statistics

THRESHOLD = 3.5
MIN_GROUPS = 3

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def median(xs):
    return statistics.median(xs) if xs else None

def mad(xs, center):
    return median([abs(x - center) for x in xs]) if xs else None

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def detect(rows, outcome="Performance-", threshold=THRESHOLD, min_groups=MIN_GROUPS):
    if not rows:
        return {"threshold": threshold, "records": [], "flagged_count": 0,
                "reason": "empty input"}
    fields = list(rows[0])
    if outcome not in fields:
        return {"threshold": threshold, "records": [], "flagged_count": 0,
                "reason": "missing outcome column"}
    features = [f for f in fields if f != outcome]
    groups = {}
    parsed = []
    for i, row in enumerate(rows):
        value = None
        try:
            candidate = float(row.get(outcome, ""))
            value = candidate if finite(candidate) else None
        except (TypeError, ValueError):
            value = None
        key = tuple(row.get(f, "") for f in features)
        groups.setdefault(key, []).append((i, value))
        parsed.append((key, value))
    keys = list(groups)
    output = []
    for i, (key, value) in enumerate(parsed):
        base = {"record_id": i + 2, "observed": value, "expected": None,
                "score": None, "threshold": threshold, "flagged": False,
                "selected_group_count": 0, "comparison_group_count": 0,
                "usable_reference_groups": [], "reason": None}
        if not finite(value):
            base["reason"] = "missing or nonfinite observed outcome converted to null"
            output.append(base); continue
        distances = [(sum(a != b for a, b in zip(key, other)), other)
                     for other in keys if other != key]
        if not distances:
            base["reason"] = "no independent configuration reference"
            output.append(base); continue
        nearest = min(d for d, _ in distances)
        refs = [k for d, k in distances if d == nearest]
        usable = [k for k in refs if any(finite(v) for _, v in groups[k])]
        base["selected_group_count"] = len(refs)
        base["usable_reference_groups"] = [dict(zip(features, k)) for k in usable]
        values = [v for k in usable for _, v in groups[k] if finite(v)]
        base["comparison_group_count"] = len(usable)
        if len(usable) < min_groups:
            base["reason"] = "insufficient independent comparison groups"
            output.append(base); continue
        if not values:
            base["reason"] = "comparison outcomes unavailable"
            output.append(base); continue
        center = median(values); spread = mad(values, center)
        base["expected"] = center
        if spread is None or spread == 0:
            if all(v == center for v in values):
                base["reason"] = "zero comparison variability"
            else:
                base["reason"] = "undefined comparison variability"
            output.append(base); continue
        base["score"] = abs(value - center) / (1.4826 * spread)
        base["flagged"] = base["score"] >= threshold
        base["reason"] = "candidate: unusually high or low relative to nearest independent groups" if base["flagged"] else "within threshold"
        output.append(base)
    return {"threshold": threshold, "records": output,
            "flagged_count": sum(r["flagged"] for r in output),
            "reason": "flags are investigation candidates, not confirmed errors"}

def main(path):
    result = detect(read_csv(path))
    print(json.dumps(result, allow_nan=False, separators=(",", ":")))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    main(p.parse_args().csv_path)
