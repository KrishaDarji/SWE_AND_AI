#!/usr/bin/env python3
"""Retrospective one-setting repair for X264 measurements."""
import csv
import math
import sys

OUTCOME = "Performance-"

def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is missing or nonfinite")
    return result

def load_records(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset has no data records")
    fields = list(rows[0])
    if OUTCOME not in fields:
        raise ValueError(f"missing outcome column {OUTCOME!r}")
    settings = [field for field in fields if field != OUTCOME]
    for index, row in enumerate(rows, 1):
        number(row[OUTCOME], f"record {index} {OUTCOME}")
        for field in settings:
            number(row[field], f"record {index} {field}")
    return rows, settings

def repair(baseline, records, settings):
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be a mapping")
    missing = [field for field in settings + [OUTCOME] if field not in baseline]
    if missing:
        raise ValueError("baseline missing columns: " + ", ".join(missing))
    base_values = {field: number(baseline[field], f"baseline {field}")
                   for field in settings + [OUTCOME]}
    candidates = []
    for order, row in enumerate(records, 1):
        outcome = number(row[OUTCOME], f"record {order} {OUTCOME}")
        if outcome >= base_values[OUTCOME]:
            continue
        changed = [field for field in settings
                   if number(row[field], f"record {order} {field}") != base_values[field]]
        candidates.append((len(changed), outcome, order, row, changed))
    if not candidates:
        return {"repaired": False, "baseline": dict(baseline)}
    _, outcome, order, row, changed = min(candidates, key=lambda item: item[:3])
    return {"repaired": True, "baseline": dict(baseline), "record_id": order,
            "repaired_config": dict(row), "changed": changed,
            "change_count": len(changed), "improvement": base_values[OUTCOME] - outcome}

def format_result(result, settings):
    lines = ["Record identifiers: one-based CSV data-record order (header excluded).",
             "Raw-column change count is a proxy; correlated/one-hot fields may represent fewer conceptual changes.",
             "Baseline: " + repr(result["baseline"])]
    if not result["repaired"]:
        lines.append("No repair: no observed record has strictly lower Performance-.")
        return "\n".join(lines)
    row = result["repaired_config"]
    lines += [f"Repaired record: {result['record_id']}", "Repaired configuration: " + repr(row),
              f"Baseline Performance-: {result['baseline'][OUTCOME]}",
              f"Repaired Performance-: {row[OUTCOME]}",
              "Changed settings: " + repr([(field, result["baseline"][field], row[field]) for field in result["changed"]]),
              f"Change count: {result['change_count']}", f"Improvement (lower is better): {result['improvement']}",
              "Observed improvement is not causal and is not a guarantee of future performance."]
    return "\n".join(lines)

def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: repair.py DATASET")
    records, settings = load_records(argv[1])
    print(format_result(repair(records[0], records, settings), settings))

if __name__ == "__main__":
    try:
        main(sys.argv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
