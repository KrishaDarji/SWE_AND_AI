#!/usr/bin/env python3
"""Marginal robust screen for unusual NASA93 outcomes."""
import argparse, csv, json, math, statistics, sys

OUTCOMES = ("Kloc+", "Effort-", "Defects-")

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def quantile(xs, p):
    if not xs: return None
    ys = sorted(xs); pos = (len(ys) - 1) * p; lo = math.floor(pos); hi = math.ceil(pos)
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)

def robust_reference(values):
    """Reference on valid records: median and MAD of log1p(value), threshold 3."""
    valid = [v for v in values if finite(v) and v >= 0]
    if not valid:
        return {"n": 0, "median": None, "mad": None, "scale": None, "threshold": None,
                "reason": "no finite nonnegative values"}
    z = [math.log1p(v) for v in valid]; med = statistics.median(z)
    mad = statistics.median([abs(x - med) for x in z]); scale = 1.4826 * mad
    return {"n": len(valid), "median": med, "mad": mad, "scale": scale,
            "threshold": 3.0, "reason": None if scale > 0 else "zero robust scale"}

def screen(records, threshold=3.0):
    refs = {}
    for name in OUTCOMES:
        refs[name] = robust_reference([r.get(name) for r in records])
        refs[name]["threshold"] = threshold
    flagged = []
    counts = {name: 0 for name in OUTCOMES}
    for row_index, row in enumerate(records):
        fields = []; reasons = []
        for name in OUTCOMES:
            value = row.get(name); ref = refs[name]
            if not finite(value) or value < 0:
                reasons.append(name + ": invalid (missing/nonfinite/negative)"); continue
            score = None if ref["scale"] in (None, 0) else abs(math.log1p(value) - ref["median"]) / ref["scale"]
            if score is not None and score > threshold:
                fields.append({"field": name, "observed": value, "score": score,
                               "reference_median_log1p": ref["median"], "reference_scale": ref["scale"],
                               "threshold": threshold})
                counts[name] += 1
        if fields or reasons:
            flagged.append({"row_index": row_index, "identifier": row.get("idX"),
                            "observed": {k: row.get(k) for k in OUTCOMES},
                            "flagged_fields": fields, "reasons": reasons})
    return {"method": "log1p robust z-score using median and 1.4826*MAD", "reference_includes_assessed_record": True,
            "threshold_rule": "flag score > threshold; zero scale yields no score-based flags",
            "references": refs, "records": flagged, "unique_flagged_record_count": len(flagged),
            "per_outcome_flag_counts": counts}

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        out = []
        for row in csv.DictReader(fh):
            item = {"idX": row.get("idX")}
            for name in OUTCOMES:
                raw = row.get(name, "").strip()
                try: item[name] = float(raw) if raw else None
                except ValueError: item[name] = None
            out.append(item)
        return out

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("csv_path"); ap.add_argument("--output", default="-")
    args = ap.parse_args(argv); result = screen(read_csv(args.csv_path))
    text = json.dumps(result, allow_nan=False, indent=2, sort_keys=True)
    if args.output == "-": print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as fh: fh.write(text + "\n")

if __name__ == "__main__": main()
