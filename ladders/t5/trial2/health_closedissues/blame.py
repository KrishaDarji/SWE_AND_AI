#!/usr/bin/env python3
"""Rank configuration settings by their marginal association with MRE-."""
import csv, math, sys
from collections import defaultdict

TARGET = "MRE-"
SETTINGS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}
EXCLUDED = {"MRE-", "ACC+", "PRED40+"}

def finite(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def mean(xs): return sum(xs) / len(xs) if xs else None
def median(xs):
    if not xs: return None
    ys = sorted(xs); n = len(ys); m = n // 2
    return ys[m] if n % 2 else (ys[m-1] + ys[m]) / 2

def variance(xs):
    if len(xs) < 2: return 0.0
    u = mean(xs)
    return sum((x-u)**2 for x in xs) / len(xs)

def fmt(x): return "NA" if x is None else f"{x:.8g}"

def quantile_groups(values, max_groups=5):
    """Make up to five equal-frequency bins from x alone; no outcomes are used."""
    ordered = sorted(values)
    distinct = sorted(set(ordered))
    k = min(max_groups, len(distinct), len(values))
    if k <= 1: return [0 for _ in values], [(values[0], values[0])] if values else []
    bounds = []
    groups = []
    for i in range(k):
        lo_i = (i * len(ordered)) // k
        hi_i = ((i + 1) * len(ordered)) // k - 1
        lo, hi = ordered[lo_i], ordered[hi_i]
        if i == k - 1: hi = ordered[-1]
        bounds.append((lo, hi))
    for x in values:
        g = min(k-1, next((i for i, (_, hi) in enumerate(bounds) if x <= hi), k-1))
        groups.append(g)
    return groups, bounds

def eta_squared(groups, ys):
    if len(ys) < 2 or len(set(groups)) < 2 or variance(ys) == 0: return 0.0
    overall = mean(ys)
    by = defaultdict(list)
    for g, y in zip(groups, ys): by[g].append(y)
    between = sum(len(v) * (mean(v)-overall)**2 for v in by.values())
    return between / (len(ys) * variance(ys))

def summarize(groups, ys):
    by = defaultdict(list)
    for g, y in zip(groups, ys): by[g].append(y)
    return [(g, len(by[g]), mean(by[g]), median(by[g]), min(by[g]), max(by[g])) for g in sorted(by, key=str)]

def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = []
    for name in SETTINGS:
        pairs = []
        for row in rows:
            y, x = finite(row.get(TARGET)), row.get(name, "").strip()
            if y is None or x in ("", "?", "NA", "nan", "None"): continue
            if name in NUMERIC:
                x = finite(x)
                if x is None: continue
            pairs.append((x, y))
        if not pairs:
            result.append({"setting": name, "score": 0.0, "n": 0, "groups": [], "direction": "insufficient data", "boundaries": []})
            continue
        xs, ys = zip(*pairs)
        if name in NUMERIC:
            gs, bounds = quantile_groups(list(xs))
            labels = [f"bin{i+1}" for i in range(len(bounds))]
            groups = [labels[g] for g in gs]
            group_ids = gs
            boundaries = [f"{fmt(a)}..{fmt(b)}" for a,b in bounds]
        else:
            # Preserve observed categorical labels; no alphabetical order is interpreted as rank.
            labels = list(dict.fromkeys(xs))
            groups = list(xs)
            group_ids = groups
            boundaries = []
        score = eta_squared(group_ids, list(ys))
        summaries = summarize(groups, list(ys))
        means = {str(g): m for g, _, m, _, _, _ in summaries}
        lo, hi = min(means.values()), max(means.values())
        direction = "lower MRE- in lower-valued groups" if means.get(str(groups[0])) == lo else "group means vary; see summaries"
        result.append({"setting": name, "score": score, "n": len(ys), "groups": summaries, "direction": direction, "boundaries": boundaries, "levels": labels})
    result.sort(key=lambda r: (-r["score"], SETTINGS.index(r["setting"])))
    return result

def main(argv):
    if len(argv) != 2:
        print("usage: blame.py DATASET.csv", file=sys.stderr); return 2
    results = analyze(argv[1])
    print("MRE- summary: lower is better")
    print("Rank | setting | eta_squared | n | groups")
    for i, r in enumerate(results, 1):
        print(f"{i} | {r['setting']} | {r['score']:.6f} | {r['n']} | {len(r['groups'])}")
        if r["boundaries"]: print("  boundaries: " + ", ".join(r["boundaries"]))
        for g, n, m, med, lo, hi in r["groups"]:
            print(f"  {g}: n={n} mean={fmt(m)} median={fmt(med)} min={fmt(lo)} max={fmt(hi)}")
    print("Method: marginal eta-squared from outcome-independent groups; no adjusted model.")
    print("Limitations: association is not causation; bins lose within-bin information; missing rows are omitted per setting; correlated settings may share information.")
    return 0

if __name__ == "__main__": sys.exit(main(sys.argv))
