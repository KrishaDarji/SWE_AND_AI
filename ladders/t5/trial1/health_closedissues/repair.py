"""Retrospective observed-data hyperparameter repair."""
import csv, math

FIELDS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}

def _number(value):
    if value is None or value == "": return None
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError): return None

def _value(field, value):
    if field in NUMERIC:
        x = _number(value)
        return x
    if field == "criterion":
        if value is None: return None
        value = str(value).strip()
        return None if not value or value == "?" else value
    return None if value is None else str(value)

def _valid_config(row):
    return all(_value(f, row.get(f)) is not None for f in FIELDS)

def _valid_outcome(row, name):
    return _number(row.get(name))

def _identifier(row, index):
    return row.get("id") or row.get("ID") or row.get("record_id") or f"csv_row_{index + 2}"

def _changes(base, candidate):
    return {f: {"old": _value(f, base.get(f)), "new": _value(f, candidate.get(f))}
            for f in FIELDS if _value(f, base.get(f)) != _value(f, candidate.get(f))}

def select(rows, baseline_index=0):
    """Select the smallest observed configuration change improving MRE-.

    Eligibility requires a valid baseline configuration/MRE- and a valid
    candidate configuration and finite MRE- strictly lower than baseline.
    """
    rows = list(rows)
    if not rows: return {"status": "empty_input"}
    if baseline_index < 0 or baseline_index >= len(rows): return {"status": "invalid_baseline", "reason": "baseline index is outside input"}
    base = rows[baseline_index]
    if not _valid_config(base): return {"status": "invalid_baseline", "reason": "baseline configuration has missing or nonfinite values"}
    bmre = _valid_outcome(base, "MRE-")
    if bmre is None: return {"status": "invalid_baseline", "reason": "baseline MRE- is missing or nonfinite"}
    eligible = []
    for i, row in enumerate(rows):
        cmre = _valid_outcome(row, "MRE-")
        if not _valid_config(row) or cmre is None or not cmre < bmre: continue
        changes = _changes(base, row)
        eligible.append((len(changes), cmre, -(_valid_outcome(row, "ACC+") if _valid_outcome(row, "ACC+") is not None else -math.inf), -(_valid_outcome(row, "PRED40+") if _valid_outcome(row, "PRED40+") is not None else -math.inf), i, row, changes))
    if not eligible: return {"status": "no_improvement", "baseline_id": _identifier(base, baseline_index), "baseline": dict(base)}
    _, cmre, _, _, i, chosen, changes = min(eligible, key=lambda x: x[:5])
    secondary = {}
    for name in OUTCOMES[1:]:
        bv, cv = _valid_outcome(base, name), _valid_outcome(chosen, name)
        secondary[name] = {"difference": None if bv is None or cv is None else cv - bv,
                           "worsened": None if bv is None or cv is None else cv < bv,
                           "reason": "unknown comparison because baseline or candidate outcome is unavailable" if bv is None or cv is None else None}
    return {"status": "improvement_found", "baseline_id": _identifier(base, baseline_index), "candidate_id": _identifier(chosen, i), "baseline": dict(base), "candidate": dict(chosen), "changes": changes, "change_count": len(changes), "change_count_is_proxy": True, "outcomes": {n: {"baseline": _valid_outcome(base, n), "candidate": _valid_outcome(chosen, n), "difference": None if _valid_outcome(base, n) is None or _valid_outcome(chosen, n) is None else _valid_outcome(chosen, n) - _valid_outcome(base, n)} for n in OUTCOMES}, "secondary": secondary, "mre_improvement": bmre - cmre}

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh: return list(csv.DictReader(fh))

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(); p.add_argument("csv_path"); args = p.parse_args()
    print(json.dumps(select(read_csv(args.csv_path)), indent=2, sort_keys=True))
