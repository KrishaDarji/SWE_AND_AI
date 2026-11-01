#!/usr/bin/env python3
"""Rank marginal configuration associations with MRE- using stdlib only."""
import csv, math, sys
from collections import defaultdict
from summarize_runs import CONFIG, NUMERIC, MISSING, finite_number

def _category(value):
    if value is None or value.strip().lower() in MISSING:
        return None
    return value.strip()

def _numeric_bins(values):
    """Return rank-balanced bins and boundaries, defined before outcomes are read."""
    ordered = sorted(set(values))
    if len(ordered) < 2:
        return None, []
    k = min(3, len(ordered))
    cuts = []
    for i in range(1, k):
        left = ordered[(i * len(ordered) + k - 1) // k - 1]
        right = ordered[(i * len(ordered) + k - 1) // k]
        if left != right:
            cuts.append((left + right) / 2)
    def label(x):
        index = sum(x > cut for cut in cuts)
        lo = ordered[0] if index == 0 else cuts[index - 1]
        hi = ordered[-1] if index == len(cuts) else cuts[index]
        return index, ("[{}..{}]".format(lo, hi))
    return label, cuts

def _score(groups):
    values = [y for group in groups.values() for y in group]
    n = len(values)
    if n < 2: return {"defined": False, "reason": "fewer than two eligible rows", "eligible": n}
    mean = sum(values) / n
    total = sum((y - mean) ** 2 for y in values)
    if total == 0: return {"defined": False, "reason": "constant MRE- outcome", "eligible": n, "reference_mean": mean, "total_ss": 0.0}
    between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
    return {"defined": True, "score": between / total, "eligible": n, "reference_mean": mean, "total_ss": total, "between_ss": between}

def _field(rows, field):
    prepared = []
    setting_values = []
    for row in rows:
        y = finite_number(row.get("MRE-", ""))
        if y is None: continue
        x = _category(row.get(field, "")) if field == "criterion" else finite_number(row.get(field, ""))
        if x is None: continue
        prepared.append((x, y)); setting_values.append(x)
    labeler, cuts = (None, []) if field == "criterion" else _numeric_bins(setting_values)
    groups = defaultdict(list)
    labels = {}
    for x, y in prepared:
        if field == "criterion": key = ("category", x); labels[key] = x
        else:
            if labeler is None:
                groups[("constant", x)].append(y); labels[("constant", x)] = "[{}..{}]".format(x, x); continue
            index, text = labeler(x); key = ("bin", index); labels[key] = text
        groups[key].append(y)
    result = _score(groups)
    result.update({"field": field, "excluded": len(rows) - len(prepared), "group_count": len(groups),
                   "groups": [{"group": labels[k], "count": len(groups[k]), "mean_mre_minus": sum(groups[k]) / len(groups[k])} for k in sorted(groups)],
                   "boundaries": cuts if field != "criterion" else None,
                   "grouping": "categorical levels" if field == "criterion" else "three rank-balanced bins (fewer when fewer distinct values)"})
    if len(groups) < 2: result.update(defined=False, reason="constant or insufficient setting groups")
    return result

def rank_associations(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    fields = [_field(rows, field) for field in CONFIG]
    defined = [x for x in fields if x.get("defined")]
    ranked = sorted(defined, key=lambda x: (-x["score"], x["field"]))
    for i, item in enumerate(ranked, 1): item["rank"] = i
    return {"outcome": "MRE-", "direction": "lower is better", "score_definition": "marginal eta-squared = between-group SS / eligible-row total SS", "fields": fields, "ranking": ranked, "excluded_predictors": ["ACC+", "PRED40+"]}

def render(result):
    lines = ["Marginal historical associations with MRE-", result["score_definition"], "Ranking (descending score; ties by field name):"]
    for x in result["ranking"]: lines.append("  {}: score={:.6g}, groups={}".format(x["field"], x["score"], x["groups"]))
    for x in result["fields"]:
        lines.append("{}: eligible={}, excluded={}, defined={}, groups={}, means={}, boundaries={}".format(x["field"], x["eligible"], x["excluded"], x["defined"], x["group_count"], x["groups"], x["boundaries"]))
    lines.append("Numeric grouping loses within-bin ordering and exact distances; criterion remains categorical. Missing/nonfinite settings or MRE- are excluded per field, so reference means and sums of squares use exactly each field's eligible rows.")
    lines.append("These are marginal historical associations; correlated settings, unequal groups, and other configuration differences affect them. They do not establish causation, significance, independent importance, or guaranteed improvement.")
    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) != 2: raise SystemExit("usage: blame.py DATASET.csv")
    print(render(rank_associations(sys.argv[1])))
