"""Retrospective configuration selector and historical baseline comparison."""
import csv
import math
import random
import statistics

FIELDS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")
EPS = 1e-12

def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def eligible_rows(rows, cap=100):
    if not _finite(cap) or float(cap) < 0:
        raise ValueError("cap must be finite and nonnegative")
    cap = float(cap)
    valid = []
    for number, raw in enumerate(rows, 2):
        if not all(raw.get(k) not in (None, "") for k in FIELDS + OUTCOMES):
            continue
        if raw.get("criterion") == "":
            continue
        try:
            parsed = dict(raw)
            parsed["N_estimators"] = float(raw["N_estimators"])
            parsed["Min_sample_leaves"] = float(raw["Min_sample_leaves"])
            parsed["Min_impurity_decrease"] = float(raw["Min_impurity_decrease"])
            parsed["Max_depth"] = float(raw["Max_depth"])
            for key in OUTCOMES:
                parsed[key] = float(raw[key])
        except (TypeError, ValueError):
            continue
        if not all(_finite(parsed[k]) for k in FIELDS if k != "criterion"):
            continue
        if not all(_finite(parsed[k]) for k in OUTCOMES):
            continue
        if parsed["N_estimators"] > cap:
            continue
        parsed["original_record"] = number
        valid.append(parsed)
    return valid

def select(rows, cap=100):
    feasible = eligible_rows(rows, cap)
    if not feasible:
        raise ValueError("no eligible records")
    return min(feasible, key=lambda r: (r["MRE-"], -r["ACC+"], -r["PRED40+"], r["original_record"]))

def baseline(rows, cap=100, seed=42):
    feasible = eligible_rows(rows, cap)
    if not feasible:
        raise ValueError("no eligible records")
    return random.Random(seed).choice(feasible)

def compare(rows, seeds=range(42, 52), cap=100):
    feasible = eligible_rows(rows, cap)
    chosen = select(rows, cap)
    records = []
    for seed in seeds:
        base = random.Random(seed).choice(feasible)
        records.append({"seed": seed, "selector": chosen, "baseline": base,
                        "difference": base["MRE-"] - chosen["MRE-"]})
    differences = [r["difference"] for r in records]
    wins = sum(d > EPS for d in differences)
    ties = sum(abs(d) <= EPS for d in differences)
    losses = sum(d < -EPS for d in differences)
    return {"records": records, "mean_difference": statistics.mean(differences),
            "median_difference": statistics.median(differences),
            "wins": wins, "ties": ties, "losses": losses}

def load(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def counts(rows, cap=100):
    valid = []
    for raw in rows:
        if all(raw.get(k) not in (None, "") for k in FIELDS + OUTCOMES):
            try:
                if all(_finite(raw[k]) for k in FIELDS if k != "criterion") and all(_finite(raw[k]) for k in OUTCOMES):
                    valid.append(raw)
            except KeyError:
                pass
    return len(rows), len(valid), len(eligible_rows(rows, cap))
