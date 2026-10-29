"""Retrospective observed-data repair by the smallest configuration change."""
import csv
import math
from pathlib import Path

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth"]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]
NUMERIC_FIELDS = set(CONFIGURATION_FIELDS) - {"criterion"}


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _configuration(row):
    return {field: row.get(field) for field in CONFIGURATION_FIELDS}


def _valid_configuration(row):
    return all(str(row.get(field, "")).strip() for field in CONFIGURATION_FIELDS) and all(
        _number(row.get(field)) is not None for field in NUMERIC_FIELDS)


def _same_setting(left, right, field):
    if field in NUMERIC_FIELDS:
        a, b = _number(left), _number(right)
        return a is not None and b is not None and a == b
    return str(left).strip() == str(right).strip()


def _outcomes(row):
    return {field: _number(row.get(field)) for field in OUTCOME_FIELDS}


def repair(path, baseline=None):
    """Find the best strictly improving observed row for a baseline.

    The change count is a configuration-field difference proxy, not measured
    implementation cost.  No prediction or causal claim is made.
    """
    rows = _rows(path)
    result = {"status": None, "record_count": len(rows), "baseline": None,
              "candidate": None, "interpretation": "retrospective observed-data repair",
              "change_count_definition": "number of differing configuration fields (proxy, not measured implementation cost)"}
    if not rows:
        result["status"] = "empty input"
        return result
    if baseline is None:
        index = 0
    elif isinstance(baseline, int) and not isinstance(baseline, bool):
        index = baseline
    elif isinstance(baseline, dict):
        index = None
        for i, row in enumerate(rows):
            if row is baseline or all(row.get(k) == baseline.get(k) for k in CONFIGURATION_FIELDS):
                index = i
                break
        if index is None:
            result["status"] = "invalid baseline: record not found"
            return result
    else:
        result["status"] = "invalid baseline: expected a CSV index or observed record"
        return result
    if index < 0 or index >= len(rows):
        result["status"] = "invalid baseline: index out of range"
        return result
    base = rows[index]
    base_out = _outcomes(base)
    result["baseline"] = {"original_record_id": index, "configuration": _configuration(base),
                           "measured_outcomes": base_out}
    if not _valid_configuration(base) or base_out["MRE-"] is None:
        result["status"] = "invalid baseline: finite MRE- and complete configuration required"
        return result
    candidates = []
    for candidate_index, row in enumerate(rows):
        values = _outcomes(row)
        if candidate_index == index or not _valid_configuration(row) or values["MRE-"] is None:
            continue
        if values["MRE-"] >= base_out["MRE-"]:
            continue
        changes = [{"field": field, "old": base.get(field), "new": row.get(field)}
                   for field in CONFIGURATION_FIELDS if not _same_setting(base.get(field), row.get(field), field)]
        candidates.append((len(changes), values["MRE-"], -(values["ACC+"] if values["ACC+"] is not None else -math.inf),
                          -(values["PRED40+"] if values["PRED40+"] is not None else -math.inf), candidate_index, row, values, changes))
    if not candidates:
        result["status"] = "no improving candidate"
        return result
    _, _, _, _, candidate_index, row, values, changes = min(candidates, key=lambda x: x[:5])
    secondary = {}
    for field in ("ACC+", "PRED40+"):
        if base_out[field] is None:
            secondary[field] = {"difference": None, "worsening": None,
                                "reason": "baseline secondary outcome unavailable; comparison unknown"}
        else:
            difference = values[field] - base_out[field] if values[field] is not None else None
            secondary[field] = {"difference": difference,
                                "worsening": difference < 0 if difference is not None else None,
                                "reason": None if difference is not None else "candidate secondary outcome unavailable"}
    result["status"] = "improving candidate found"
    result["candidate"] = {"original_record_id": candidate_index, "configuration": _configuration(row),
                            "measured_outcomes": values, "changes": changes, "change_count": len(changes),
                            "mre_improvement": base_out["MRE-"] - values["MRE-"], "secondary_comparisons": secondary}
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(repair(args.dataset), indent=2, sort_keys=True))
