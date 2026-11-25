"""Flag unusual X264 Performance- values using independent nearest configurations."""
import csv, json, math, sys

TARGET = "Performance-"
DEFAULT_THRESHOLD = 3.0
MIN_COMPARISONS = 2

def _num(value):
    try:
        x = float(str(value).strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def _setting(value):
    value = "" if value is None else str(value).strip()
    return None if value in ("", "?") else value

def _median(values):
    values = sorted(values)
    if not values: return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2

def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if TARGET not in fields: raise ValueError("dataset must contain Performance-")
        settings = [f for f in fields if f != TARGET]
        rows = []
        for number, row in enumerate(reader, 1):
            outcome = _num(row.get(TARGET))
            if outcome is not None:
                rows.append({"record_id": number, "row": row, "outcome": outcome,
                             "configuration": tuple(_setting(row.get(f)) for f in settings)})
        return settings, rows

def _score(observed, references):
    center = sum(references) / len(references)
    deviations = [abs(x - center) for x in references]
    scale = 1.4826 * _median(deviations)
    if scale == 0:
        spread = math.sqrt(sum((x - center) ** 2 for x in references) / len(references))
        scale = spread
    if scale == 0: return None, center, "zero reference variability"
    return abs(observed - center) / scale, center, None

def spot(path, threshold=DEFAULT_THRESHOLD):
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be nonnegative and finite")
    settings, rows = _read(path)
    results = []
    for current in rows:
        distances = []
        for other in rows:
            if other is current: continue
            distance = sum(a != b for a, b in zip(current["configuration"], other["configuration"]))
            if distance: distances.append((distance, other))
        minimum = min((d for d, _ in distances), default=None)
        peers = [other for d, other in distances if d == minimum] if minimum is not None else []
        references = [peer["outcome"] for peer in peers]
        score, expected, reason = (None, None, "no independent comparison group") if len(references) < MIN_COMPARISONS else _score(current["outcome"], references)
        if reason is None and score is not None and score >= threshold: reason = "score meets threshold"
        results.append({"record_id": current["record_id"], "observed": current["outcome"],
                        "expected": expected, "score": score, "threshold": threshold,
                        "flagged": bool(score is not None and score >= threshold),
                        "reason": reason or "within threshold", "comparison_distance": minimum,
                        "comparison_group_count": len(peers),
                        "comparison_record_ids": [peer["record_id"] for peer in peers]})
    return {"target": TARGET, "direction": "lower Performance- is better",
            "method": "For each record, use all finite records at the smallest nonzero Hamming distance across settings; identical configurations are excluded.",
            "score_definition": "absolute observed-minus-reference mean divided by 1.4826*MAD of independent references, with RMS fallback",
            "threshold": threshold, "minimum_comparisons": MIN_COMPARISONS,
            "settings": settings, "rows_with_finite_outcome": len(rows),
            "records": results, "flagged_record_count": sum(x["flagged"] for x in results),
            "limitations": ["Flags are investigation candidates, not confirmed measurement errors.", "Nearest observational comparisons may be confounded and are unavailable for tiny or identical-only inputs.", "No detection accuracy is claimed without verified anomaly labels."]}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1: print("usage: x264_spot.py DATASET", file=sys.stderr); return 2
    print(json.dumps(spot(argv[0]), indent=2, sort_keys=True, allow_nan=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
