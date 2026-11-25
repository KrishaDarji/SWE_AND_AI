"""Retrospective observed-data hyperparameter repair."""
import csv, math

CONFIG = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
NUMERIC = {"N_estimators", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"}
OUTCOMES = ("MRE-", "ACC+", "PRED40+")

def _number(value, field):
    if value is None or str(value).strip() == "":
        raise ValueError("missing " + field)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid " + field) from exc
    if not math.isfinite(result):
        raise ValueError("nonfinite " + field)
    if field in ("N_estimators", "Min_sample_leaves", "Max_depth") and not result.is_integer():
        raise ValueError("non-integral " + field)
    return int(result) if field in ("N_estimators", "Min_sample_leaves", "Max_depth") else result

def _record(raw, identifier):
    row = {key: raw.get(key) for key in raw}
    row["MRE-"] = _number(raw.get("MRE-"), "MRE-")
    for key in ("ACC+", "PRED40+"):
        row[key] = None if raw.get(key) in (None, "") else _number(raw.get(key), key)
    for key in CONFIG:
        if key == "criterion":
            if not raw.get(key): raise ValueError("missing criterion")
            row[key] = raw[key]
        else: row[key] = _number(raw.get(key), key)
    row["identifier"] = identifier
    return row

def _diff(a, b, key):
    return a[key] != b[key]

def choose(rows):
    """Choose the least-changing observed row with strictly lower MRE-."""
    if not rows: raise ValueError("empty input")
    baseline = rows[0]
    try: base = _record(baseline, baseline.get("identifier", 1))
    except ValueError as exc: raise ValueError("invalid baseline: " + str(exc)) from exc
    candidates = []
    for position, raw in enumerate(rows[1:], 2):
        try: rec = _record(raw, raw.get("identifier", position))
        except ValueError: continue
        if rec["MRE-"] < base["MRE-"]:
            changes = [{"field": key, "old": base[key], "new": rec[key]} for key in CONFIG if _diff(base, rec, key)]
            candidates.append((len(changes), rec["MRE-"], -(rec["ACC+"] if rec["ACC+"] is not None else -math.inf), -(rec["PRED40+"] if rec["PRED40+"] is not None else -math.inf), position, rec, changes))
    if not candidates:
        return {"status": "no_improvement", "baseline": base, "candidate": None, "reason": "no usable candidate has strictly lower observed MRE-"}
    _, _, _, _, _, best, changes = min(candidates, key=lambda x: x[:5])
    result = {"status": "improved", "baseline": base, "candidate": best, "changes": changes,
              "change_count": len(changes), "mre_improvement": base["MRE-"] - best["MRE-"]}
    for key in ("ACC+", "PRED40+"):
        if base[key] is None or best[key] is None:
            result[key + "_difference"] = None
            result[key + "_worsened"] = None
            result[key + "_unknown_reason"] = "baseline or candidate secondary outcome unavailable"
        else:
            result[key + "_difference"] = best[key] - base[key]
            result[key + "_worsened"] = best[key] < base[key]
    return result

def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [{**row, "identifier": index} for index, row in enumerate(csv.DictReader(handle), 1)]

def main(path):
    result = choose(load_csv(path))
    print("status:", result["status"])
    print("baseline_identifier:", result["baseline"]["identifier"])
    print("baseline_configuration:", {key: result["baseline"][key] for key in CONFIG})
    if result["candidate"] is None: print("reason:", result["reason"]); return result
    print("candidate_identifier:", result["candidate"]["identifier"])
    print("candidate_configuration:", {key: result["candidate"][key] for key in CONFIG})
    print("outcomes:", {key: (result["baseline"][key], result["candidate"][key]) for key in OUTCOMES})
    print("changes:", result["changes"])
    print("change_count_proxy:", result["change_count"])
    print("observed_MRE-_improvement:", result["mre_improvement"])
    print("ACC+_difference:", result["ACC+_difference"], "worsened:", result["ACC+_worsened"])
    print("PRED40+_difference:", result["PRED40+_difference"], "worsened:", result["PRED40+_worsened"])
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2: raise SystemExit("usage: python repair.py DATASET.csv")
    main(sys.argv[1])
