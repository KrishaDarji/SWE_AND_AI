#!/usr/bin/env python3
"""Rank settings by their marginal association with lower-is-better performance."""
import csv, math, statistics, sys

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows

def summarize(values):
    vals = [v for v in values if finite(v)]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(vals), "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}

def rank_settings(rows, outcome="Performance-"):
    settings = [k for k in rows[0] if k != outcome]
    result = []
    overall = [float(r[outcome]) for r in rows
               if r.get(outcome, "") not in ("") and finite(float(r[outcome]))]
    overall_mean = statistics.fmean(overall)
    total_ss = sum((y - overall_mean) ** 2 for y in overall)
    for name in settings:
        groups = {}
        missing = []
        for r in rows:
            raw = r.get(name, "")
            try:
                x = float(raw)
                y = float(r[outcome])
                if not finite(x) or not finite(y):
                    raise ValueError
            except (ValueError, TypeError):
                missing.append(r)
                continue
            groups.setdefault(x, []).append(y)
        between = sum(len(v) * (statistics.fmean(v) - overall_mean) ** 2
                      for v in groups.values())
        score = between / total_ss if total_ss else 0.0
        ordered = sorted(groups.items(), key=lambda item: item[0])
        summaries = [{"value": value, **summarize(vals)} for value, vals in ordered]
        direction = None
        if len(ordered) == 2:
            direction = ordered[1][1] and statistics.fmean(ordered[1][1]) - statistics.fmean(ordered[0][1])
        result.append({"setting": name, "score": score, "groups": len(groups),
                       "missing_count": len(missing), "direction_high_minus_low": direction,
                       "group_summaries": summaries})
    result.sort(key=lambda x: (-x["score"], x["setting"]))
    for i, item in enumerate(result, 1):
        item["rank"] = i
    return result

def main(path):
    rows = read_csv(path)
    ranked = rank_settings(rows)
    print("rows=%d outcome=Performance- lower_is_better=True" % len(rows))
    print("rank,setting,eta_squared,groups,missing,group_summaries,direction_high_minus_low")
    for x in ranked:
        summaries = ";".join("%s:n=%d mean=%.6f" % (g["value"], g["n"], g["mean"])
                              for g in x["group_summaries"])
        direction = "NA" if x["direction_high_minus_low"] is None else "%.6f" % x["direction_high_minus_low"]
        print("%d,%s,%.9f,%d,%d,%s,%s" % (x["rank"], x["setting"], x["score"],
              x["groups"], x["missing_count"], summaries, direction))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python analyzer.py DATASET.csv")
    main(sys.argv[1])
