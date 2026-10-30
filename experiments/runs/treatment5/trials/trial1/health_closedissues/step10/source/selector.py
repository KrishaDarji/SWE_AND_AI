"""Retrospective configuration comparison using only the Python standard library."""
import csv
import math
import random

CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")
CAP = 100


def _number(value):
    if value is None or not isinstance(value, str) or value.strip() in ("", "?"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record(row, original_id, cap=CAP):
    if cap is not None and (not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0):
        raise ValueError("cap must be finite and nonnegative")
    criterion = row.get("criterion")
    if criterion is None or not isinstance(criterion, str) or criterion.strip() in ("", "?"):
        return None
    numeric = {}
    for key in ("N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"):
        value = _number(row.get(key))
        if value is None:
            return None
        numeric[key] = value
    outcomes = {}
    for key in OUTCOMES:
        value = _number(row.get(key))
        if value is None:
            return None
        outcomes[key] = value
    if cap is not None and numeric["N_estimators"] > cap:
        return None
    return {"original_id": original_id, "configuration": {key: str(row.get(key)).strip() for key in CONFIG},
            "numeric_configuration": numeric, "outcomes": outcomes}


def load_records(path, cap=CAP):
    """Load records, preserving the one-based original CSV data-row identifier."""
    if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    raw = valid = feasible = 0
    records = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for original_id, row in enumerate(reader, 1):
            raw += 1
            candidate = _record(row, original_id, None)
            if candidate is not None:
                valid += 1
            record = _record(row, original_id, cap)
            if record is not None:
                feasible += 1
                records.append(record)
    return records, {"raw_total": raw, "valid": valid, "feasible": feasible}


def select(records):
    if not records:
        raise ValueError("no eligible records")
    return min(records, key=lambda item: (item["outcomes"]["MRE-"], -item["outcomes"]["ACC+"],
                                          -item["outcomes"]["PRED40+"], item["original_id"]))


def compare(records, seeds=range(42, 52)):
    chosen = select(records)
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        baseline = rng.choice(records)
        difference = baseline["outcomes"]["MRE-"] - chosen["outcomes"]["MRE-"]
        rows.append({"seed": seed, "selector": chosen, "baseline": baseline, "difference": difference})
    return rows


def summarize(rows, tolerance=1e-12):
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    differences = [row["difference"] for row in rows]
    wins = sum(value > tolerance for value in differences)
    ties = sum(abs(value) <= tolerance for value in differences)
    losses = sum(value < -tolerance for value in differences)
    ordered = sorted(differences)
    median = None if not ordered else ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2
    return {"count": len(differences), "mean_difference": None if not differences else sum(differences) / len(differences),
            "median_difference": median, "wins": wins, "ties": ties, "losses": losses,
            "acc_tradeoffs": sum(row["baseline"]["outcomes"]["ACC+"] - row["selector"]["outcomes"]["ACC+"] for row in rows),
            "pred40_tradeoffs": sum(row["baseline"]["outcomes"]["PRED40+"] - row["selector"]["outcomes"]["PRED40+"] for row in rows)}
