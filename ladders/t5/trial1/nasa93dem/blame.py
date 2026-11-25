"""Rank categorical project descriptors by marginal association with Effort-."""
import csv, math, sys

EXCLUDE = {"idX", "Kloc+", "Effort-", "Defects-", "MonthsX"}

def _finite(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def _summary(values):
    n = 0; mean = 0.0; m2 = 0.0
    for value in values:
        v = _finite(value)
        if v is None: continue
        n += 1; delta = v - mean; mean += delta / n; m2 += delta * (v - mean)
    variance = m2 / (n - 1) if n > 1 else 0.0
    return {"n": n, "mean": mean if n else None, "sd": math.sqrt(max(0.0, variance))}

def _effect(a, b):
    if a["n"] < 2 or b["n"] < 2:
        return {"score": None, "reason": "insufficient observations"}
    pooled = math.sqrt(((a["n"]-1)*a["sd"]**2 + (b["n"]-1)*b["sd"]**2) /
                       max(1, a["n"] + b["n"] - 2))
    difference = a["mean"] - b["mean"]
    if pooled == 0:
        if difference == 0: return {"score": 0.0, "reason": "zero variability and equal means"}
        return {"score": math.copysign(math.inf, difference), "reason": "zero variability and different means"}
    return {"score": difference / pooled, "reason": "standardized pooled-SD effect"}

def rank_rows(rows, target="Effort-"):
    if not rows or target not in rows[0]: return []
    descriptors = [k for k in rows[0] if k not in EXCLUDE and not k.endswith("X") and k != target]
    result = []
    for name in descriptors:
        groups = {}
        for row in rows:
            key = str(row.get(name, "")).strip()
            effort = _finite(row.get(target))
            if key and key != "?" and effort is not None: groups.setdefault(key, []).append(effort)
        all_values = [v for vs in groups.values() for v in vs]
        overall = _summary(all_values)
        candidates = []
        for key in sorted(groups):
            inside = groups[key]; outside = [v for k, vs in groups.items() if k != key for v in vs]
            effect = _effect(_summary(inside), _summary(outside))
            score = effect["score"]
            rank_score = abs(score) if score is not None else 0.0
            candidates.append((rank_score, score, effect["reason"], key, len(inside), _summary(inside)))
        best = max(candidates, key=lambda x: (x[0], -len(x[3]), x[3])) if candidates else (0, 0, "no category", "", 0, _summary([]))
        signed = best[1]
        result.append({"descriptor": name, "score": best[0], "signed_score": signed, "reason": best[2],
                       "direction": "higher" if signed is not None and signed > 0 else "lower" if signed is not None and signed < 0 else "none",
                       "group": best[3], "group_n": best[4], "group_mean": best[5]["mean"],
                       "overall_n": overall["n"], "overall_mean": overall["mean"],
                       "groups": {k: {**_summary(v), "count": len(v)} for k, v in sorted(groups.items())}})
    return sorted(result, key=lambda x: (-x["score"], x["descriptor"]))

def load(path):
    with open(path, newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))

def report(rows):
    out = ["Marginal categorical association with observed Effort- (Cohen-style pooled-SD score)",
           "Score is the largest absolute category-vs-rest standardized mean gap; sign is category mean minus rest mean.",
           "descriptor,score,direction,highest_group,group_n,group_mean,overall_n,overall_mean"]
    for item in rank_rows(rows):
        score = "unbounded" if math.isinf(item["score"]) else f"{item['score']:.6f}"
        out.append(f"{item['descriptor']},{score},{item['direction']},{item['group']},{item['group_n']},{item['group_mean']},{item['overall_n']},{item['overall_mean']},{item['reason']}")
    return "\n".join(out)

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: python source/blame.py DATASET.csv")
    print(report(load(sys.argv[1])))
