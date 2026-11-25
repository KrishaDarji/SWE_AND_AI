"""Retrospective configuration selector and historical baseline comparison."""
import csv, math, random

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")

def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def eligible(row, cap=100):
    if not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    if any(row.get(k, "").strip() == "" for k in CONFIG + OUTCOMES):
        return False
    for key in OUTCOMES + ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"):
        if finite_number(row.get(key)) is None:
            return False
    if row.get("criterion", "").strip() == "":
        return False
    return float(row["N_estimators"]) <= cap

def load(path, cap=100):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if all(k in r for k in CONFIG + OUTCOMES)]
    feasible = [(i, r) for i, r in enumerate(valid) if eligible(r, cap)]
    return rows, valid, feasible

def selector(feasible):
    if not feasible:
        raise ValueError("no eligible records")
    # EZR's stable keyed ordering is represented by explicit original position.
    return min(feasible, key=lambda ir: (float(ir[1]["MRE-"]), -float(ir[1]["ACC+"]),
                                         -float(ir[1]["PRED40+"]), ir[0]))

def baseline(feasible, seed):
    if not feasible:
        raise ValueError("no eligible records")
    return random.Random(seed).choice(feasible)

def comparison(path, seeds=range(42, 52), cap=100):
    raw, valid, feasible = load(path, cap)
    chosen = selector(feasible)
    records = []
    for seed in seeds:
        base = baseline(feasible, seed)
        diff = float(base[1]["MRE-"]) - float(chosen[1]["MRE-"])
        records.append((seed, chosen, base, diff))
    diffs = [x[3] for x in records]
    wins = sum(d > 1e-12 for d in diffs)
    losses = sum(d < -1e-12 for d in diffs)
    return {"raw_total": len(raw), "valid": len(valid), "feasible": len(feasible),
            "selector": chosen, "records": records, "mean_difference": sum(diffs)/len(diffs),
            "median_difference": sorted(diffs)[len(diffs)//2] if len(diffs) % 2 else
                (sorted(diffs)[len(diffs)//2-1] + sorted(diffs)[len(diffs)//2])/2,
            "wins": wins, "ties": len(diffs)-wins-losses, "losses": losses}
