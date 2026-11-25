#!/usr/bin/env python3
"""Rank categorical NASA93 project descriptors by observed effort association."""
import csv, math, statistics, sys

EXCLUDED = {"idX", "centerX", "YearX", "Kloc+", "Effort-", "Defects-", "MonthsX"}

def clean(value):
    value = value.strip()
    return value if value and value not in {"?", "NA", "nan", "NaN"} else None

def median(values):
    return statistics.median(values) if values else None

def cliffs(xs, ys):
    if not xs or not ys:
        return None
    gt = sum(x > y for x in xs for y in ys)
    lt = sum(x < y for x in xs for y in ys)
    return (gt - lt) / (len(xs) * len(ys))

def rank_rows(rows, descriptors, target="Effort-"):
    efforts = [r[target] for r in rows if isinstance(r.get(target), (int, float)) and math.isfinite(r[target])]
    overall = statistics.mean(efforts) if efforts else None
    total_ss = sum((y - overall) ** 2 for y in efforts) if overall is not None else 0.0
    result = []
    for name in descriptors:
        groups = {}
        for row in rows:
            key, y = row.get(name), row.get(target)
            if key is not None and isinstance(y, (int, float)) and math.isfinite(y):
                groups.setdefault(key, []).append(y)
        valid = {k: v for k, v in groups.items() if v}
        between = sum(len(v) * (statistics.mean(v) - overall) ** 2 for v in valid.values()) if overall is not None else 0.0
        score = between / total_ss if total_ss > 0 else 0.0
        high = max(valid, key=lambda k: (statistics.median(valid[k]), str(k))) if valid else None
        low = min(valid, key=lambda k: (statistics.median(valid[k]), str(k))) if valid else None
        deltas = []
        keys = sorted(valid, key=str)
        for key in keys:
            others = [y for other, vals in valid.items() if other != key for y in vals]
            d = cliffs(valid[key], others)
            if d is not None:
                deltas.append(d)
        result.append({"descriptor": name, "score": score, "groups": len(valid),
                       "n": sum(map(len, valid.values())), "high": high, "high_median": median(valid[high]) if high is not None else None,
                       "low": low, "low_median": median(valid[low]) if low is not None else None,
                       "max_abs_cliffs": max((abs(d) for d in deltas), default=0.0), "group_summaries": valid})
    return sorted(result, key=lambda x: (-x["score"], x["descriptor"]))

def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    descriptors = [k for k in raw[0] if k not in EXCLUDED]
    rows = []
    for item in raw:
        row = {k: clean(v) for k, v in item.items()}
        try:
            row["Effort-"] = float(row["Effort-"]) if row["Effort-"] is not None else None
        except (TypeError, ValueError):
            row["Effort-"] = None
        rows.append(row)
    return rows, descriptors

def render(path):
    rows, descriptors = load(path)
    ranked = rank_rows(rows, descriptors)
    effort = [r["Effort-"] for r in rows if r.get("Effort-") is not None]
    print("NASA93 excessive-effort association ranking")
    print("Primary score: eta_squared = between-group effort SS / total effort SS; larger means stronger marginal association.")
    print("Rows={}, effort_n={}, effort_median={:.3f}, effort_mean={:.3f}".format(len(rows), len(effort), median(effort), statistics.mean(effort)))
    print("rank,descriptor,eta_squared,groups,n,highest_median_group,highest_median,lowest_median_group,lowest_median,max_abs_cliffs")
    for i, item in enumerate(ranked, 1):
        print("{},{},{:.6f},{},{},{},{:.3f},{},{:.3f},{:.6f}".format(i, item["descriptor"], item["score"], item["groups"], item["n"], item["high"], item["high_median"], item["low"], item["low_median"], item["max_abs_cliffs"]))
    print("Interpretation: highest-median categories are associated with higher observed effort in this sample; this is not causal or size-adjusted productivity.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: blame.py DATASET.csv")
    render(sys.argv[1])
