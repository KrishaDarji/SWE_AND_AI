#!/usr/bin/env python3
"""Rank configuration settings by their observed association with MRE-."""
import csv, math, sys
from collections import defaultdict

NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}
SETTINGS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]

def number(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def quantile_groups(values, wanted=4):
    """Outcome-free equal-frequency groups; returns labels and boundaries."""
    good = sorted(set(v for v in values if isinstance(v, (int, float)) and math.isfinite(v)))
    if not good:
        return [None] * len(values), []
    k = min(wanted, len(good))
    cuts = [good[min(len(good) - 1, math.ceil(i * len(good) / k) - 1)] for i in range(1, k)]
    labels = []
    for value in values:
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            labels.append(None)
        else:
            labels.append(sum(value > cut for cut in cuts))
    bounds = []
    lo = good[0]
    for cut in cuts:
        bounds.append((lo, cut))
        lo = math.nextafter(cut, math.inf)
    bounds.append((lo, good[-1]))
    return labels, bounds

def _numeric_direction(means):
    ordered = [means[k] for k in sorted(means)]
    if not ordered or max(ordered) == min(ordered):
        return "no variation"
    diffs = [b - a for a, b in zip(ordered, ordered[1:])]
    if all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
        return "higher MRE- in higher numeric groups"
    if all(d <= 0 for d in diffs) and any(d < 0 for d in diffs):
        return "lower MRE- in higher numeric groups"
    return "nonmonotonic"

def eta_squared(y, groups, ordered_numeric=False):
    pairs = [(a, b) for a, b in zip(y, groups) if a is not None and b is not None]
    if not pairs:
        return 0.0, 0, 0, None, None
    mean = sum(a for a, _ in pairs) / len(pairs)
    total = sum((a - mean) ** 2 for a, _ in pairs)
    buckets = defaultdict(list)
    for a, b in pairs: buckets[b].append(a)
    between = sum(len(v) * (sum(v) / len(v) - mean) ** 2 for v in buckets.values())
    score = between / total if total > 0 else 0.0
    means = {str(k): sum(v) / len(v) for k, v in buckets.items()}
    direction = _numeric_direction({int(k): v for k, v in means.items()}) if ordered_numeric else "category-specific means"
    return score, len(pairs), len(buckets), means, direction

def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = [number(r.get("MRE-")) for r in rows]
    out = []
    for name in SETTINGS:
        raw = [r.get(name, "") for r in rows]
        if name in NUMERIC:
            vals = [number(x) for x in raw]
            groups, bounds = quantile_groups(vals)
            grouping = "quantile4"
            boundary_text = ";".join("%g..%g" % b for b in bounds)
        else:
            groups = [x if x not in (None, "?") else None for x in raw]
            bounds, grouping, boundary_text = [], "levels", ""
        score, n, ng, means, direction = eta_squared(y, groups, ordered_numeric=name in NUMERIC)
        out.append({"setting": name, "score": score, "n": n, "groups": ng,
                    "missing": len(rows) - n, "grouping": grouping,
                    "boundaries": boundary_text, "means": means or {}, "direction": direction})
    out.sort(key=lambda z: (-z["score"], SETTINGS.index(z["setting"])))
    return rows, out

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: blame.py DATASET.csv")
    rows, results = analyze(argv[1])
    y = [number(r.get("MRE-")) for r in rows if number(r.get("MRE-")) is not None]
    if not y:
        print("No usable MRE- outcomes; no association scores can be computed.")
        return
    print("MRE- summary: n=%d mean=%.6g min=%.6g max=%.6g lower_is_better=yes" %
          (len(y), sum(y) / len(y), min(y), max(y)))
    print("rank,setting,eta_squared,n,groups,missing,grouping,boundaries,direction,group_means")
    for i, z in enumerate(results, 1):
        print("%d,%s,%.9f,%d,%d,%d,%s,%s,%s,%s" %
              (i, z["setting"], z["score"], z["n"], z["groups"], z["missing"],
               z["grouping"], z["boundaries"], z["direction"], repr(z["means"])))

if __name__ == "__main__": main(sys.argv)
