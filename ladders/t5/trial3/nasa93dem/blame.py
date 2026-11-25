#!/usr/bin/env python3
"""Rank categorical NASA93 descriptors by their marginal association with effort."""
import csv, math, statistics

EXCLUDED = {"Kloc+", "Effort-", "Defects-"}

def finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, skipinitialspace=False))
    if not rows:
        raise ValueError("dataset has no data rows")
    fields = list(rows[0])
    target = "Effort-"
    descriptors = [x for x in fields if x not in EXCLUDED and not x.endswith("X")]
    if target not in fields:
        raise ValueError("missing Effort- column")
    return rows, descriptors

def summarize(values):
    vals = sorted(values)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "sd": None,
                "min": None, "max": None}
    return {"n": len(vals), "mean": statistics.fmean(vals),
            "median": statistics.median(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": vals[0], "max": vals[-1]}

def association(rows, descriptor):
    groups = {}
    all_values = []
    for row in rows:
        effort = finite(row.get("Effort-"))
        value = row.get(descriptor)
        value = value.strip() if isinstance(value, str) else value
        if effort is None or value in (None, "", "?"):
            continue
        groups.setdefault(value, []).append(effort)
        all_values.append(effort)
    overall = summarize(all_values)
    if len(all_values) < 2 or len(groups) < 2:
        score = 0.0
    else:
        mean = overall["mean"]
        between = sum(len(v) * (statistics.fmean(v) - mean) ** 2
                      for v in groups.values())
        total = sum((x - mean) ** 2 for x in all_values)
        score = between / total if total > 0 else 0.0
    ordered = sorted(groups.items(), key=lambda kv: (-statistics.fmean(kv[1]), kv[0]))
    high = ordered[0][0] if ordered else None
    low = ordered[-1][0] if ordered else None
    direction = "higher effort in " + str(high) if high is not None and high != low else "none"
    return {"descriptor": descriptor, "score": score, "groups": groups,
            "overall": overall, "high": high, "low": low, "direction": direction}

def analyze(path):
    rows, descriptors = read_rows(path)
    results = [association(rows, d) for d in descriptors]
    results.sort(key=lambda r: (-r["score"], r["descriptor"]))
    return results

def report(path):
    results = analyze(path)
    print("descriptor,eta_squared,usable_n,groups,high_effort_group,low_effort_group,direction")
    for r in results:
        print("%s,%.6f,%d,%d,%s,%s,%s" %
              (r["descriptor"], r["score"], r["overall"]["n"], len(r["groups"]),
               r["high"] or "", r["low"] or "", r["direction"]))
        for group in sorted(r["groups"]):
            s = summarize(r["groups"][group])
            print("  group=%s n=%d mean=%.6f median=%.6f sd=%.6f min=%.6f max=%.6f" %
                  (group, s["n"], s["mean"], s["median"], s["sd"], s["min"], s["max"]))
    print("Interpretation: eta_squared is marginal between-group effort variation / total variation.")
    print("Higher effort is not inefficiency: scope differs; size-adjusted productivity is not measured.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    report(p.parse_args().dataset)
