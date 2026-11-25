#!/usr/bin/env python3
"""Rank configuration settings by association with the lower-is-better MRE-."""
import argparse, csv, math, os, statistics, sys

SETTINGS = ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
OUTCOME = "MRE-"

def finite(x):
    return x is not None and math.isfinite(x)

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def groups_numeric(values, k=5):
    """Equal-count rank groups; ties may make groups uneven, but never use y."""
    pairs = sorted((v, i) for i, v in enumerate(values) if finite(v))
    groups = [None] * len(values)
    n = len(pairs)
    for rank, (_, i) in enumerate(pairs):
        groups[i] = min(k - 1, rank * k // n)
    bounds = []
    for g in range(k):
        vals = [v for v, i in pairs if groups[i] == g]
        bounds.append((min(vals), max(vals), len(vals)) if vals else None)
    return groups, bounds

def association(rows, setting):
    usable = [(r[setting], r[OUTCOME]) for r in rows if finite(r[OUTCOME]) and r[setting] is not None]
    ys = [y for _, y in usable]
    overall = mean(ys)
    sst = sum((y - overall) ** 2 for y in ys)
    if not usable or sst == 0 or len(set(v for v, _ in usable)) < 2:
        return {"setting": setting, "score": 0.0, "n": len(usable), "groups": 0,
                "outcome_min": min(ys) if ys else None, "outcome_median": statistics.median(ys) if ys else None,
                "outcome_mean": overall, "outcome_max": max(ys) if ys else None, "direction": "constant/undefined",
                "details": "No variation in usable outcomes or setting."}
    if setting == "criterion":
        labels = sorted(set(v for v, _ in usable), key=str)
        keys = [v for v, _ in usable]
        bounds = None
    else:
        vals = [v for v, _ in usable]
        keys, bounds = groups_numeric(vals)
        labels = list(range(5))
    by = {g: [] for g in labels}
    for g, (_, y) in zip(keys, usable):
        if g is not None: by[g].append(y)
    sse = sum(sum((y - mean(by[g])) ** 2 for y in by[g]) for g in labels if by[g])
    score = 1.0 - sse / sst
    means = {str(g): mean(by[g]) for g in labels if by[g]}
    ordered = sorted(((m, g) for g, m in means.items()), key=lambda z: (z[0], z[1]))
    direction = "lower MRE- in lower setting/group" if len(ordered) > 1 and ordered[0][1] == str(labels[0]) else "mixed/non-monotonic"
    detail = "groups=" + repr(means)
    if bounds is not None: detail += "; bounds=" + repr(bounds)
    return {"setting": setting, "score": score, "n": len(usable), "groups": len(means),
            "outcome_min": min(ys), "outcome_median": statistics.median(ys), "outcome_mean": overall,
            "outcome_max": max(ys), "direction": direction, "details": detail}

def load(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            r = {OUTCOME: float(raw[OUTCOME]) if raw.get(OUTCOME) not in (None, "", "?") else None}
            for s in SETTINGS:
                v = raw.get(s)
                if v in (None, "", "?"): r[s] = None
                elif s == "criterion": r[s] = v
                else:
                    try: r[s] = float(v) if math.isfinite(float(v)) else None
                    except (TypeError, ValueError): r[s] = None
            rows.append(r)
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args(argv)
    rows = load(args.dataset)
    results = [association(rows, s) for s in SETTINGS]
    results.sort(key=lambda r: (-r["score"], r["setting"]))
    print("MRE- association ranking (higher score means stronger explanatory association; lower MRE- is better)")
    for rank, r in enumerate(results, 1):
        print(f"{rank}. {r['setting']}: score={r['score']:.9f}, n={r['n']}, groups={r['groups']}, "
              f"MRE- min/median/mean/max={r['outcome_min']:.6g}/{r['outcome_median']:.6g}/"
              f"{r['outcome_mean']:.6g}/{r['outcome_max']:.6g}; {r['direction']}; {r['details']}")
    with open(args.metrics, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["setting", "score", "n", "groups", "outcome_min", "outcome_median", "outcome_mean", "outcome_max", "direction", "details"])
        for r in results: w.writerow([r[k] for k in ["setting","score","n","groups","outcome_min","outcome_median","outcome_mean","outcome_max","direction","details"]])
    return 0

if __name__ == "__main__": sys.exit(main())
