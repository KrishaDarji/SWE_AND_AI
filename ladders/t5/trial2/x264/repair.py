#!/usr/bin/env python3
"""Exhaustive minimum-change retrospective repair for X264 measurements."""
import argparse
import csv
import json
import math
import sys


TARGET = "Performance-"


def finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is missing or nonnumeric: {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"{label} is nonfinite: {value!r}")
    return number


def read_measurements(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or TARGET not in reader.fieldnames:
            raise ValueError(f"CSV must contain {TARGET!r}")
        settings = [name for name in reader.fieldnames if name != TARGET]
        rows = []
        for order, row in enumerate(reader, 1):
            item = dict(row)
            item["_order"] = order
            rows.append(item)
    return settings, rows


def _numeric_equal(left, right):
    return finite_number(left, "setting") == finite_number(right, "setting")


def repair(baseline, rows, setting_columns):
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be a mapping of raw column names to values")
    if "_order" not in baseline:
        raise ValueError("baseline must include its original _order identifier")
    try:
        baseline_order = int(baseline["_order"])
    except (TypeError, ValueError):
        raise ValueError("baseline _order must be an integer")
    missing = [name for name in setting_columns if name not in baseline]
    if missing:
        raise ValueError(f"baseline is missing settings: {missing}")
    base_outcome = finite_number(baseline.get(TARGET), "baseline outcome")
    for name in setting_columns:
        finite_number(baseline[name], f"baseline setting {name}")
    candidates = []
    for row in rows:
        outcome = finite_number(row.get(TARGET), f"record {row.get('_order')} outcome")
        if not outcome < base_outcome:
            continue
        changed = []
        for name in setting_columns:
            if not _numeric_equal(baseline[name], row.get(name)):
                changed.append({"setting": name, "old": baseline[name], "new": row.get(name)})
        candidates.append((len(changed), outcome, row["_order"], row, changed))
    if not candidates:
        return {"repaired": False, "reason": "no observed improvement",
                "baseline_identifier": f"record-{baseline_order}",
                "baseline_outcome": base_outcome}
    _, outcome, order, row, changed = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "repaired": True,
        "baseline_identifier": f"record-{baseline_order}",
        "repaired_identifier": f"record-{order}",
        "baseline": {name: baseline[name] for name in setting_columns + [TARGET]},
        "repaired_configuration": {name: row[name] for name in setting_columns + [TARGET]},
        "changed_settings": changed,
        "change_count": len(changed),
        "improvement": base_outcome - outcome,
        "baseline_outcome": base_outcome,
        "repaired_outcome": outcome,
        "raw_column_change_count_proxy": "correlated or one-hot fields may represent fewer conceptual changes",
        "caveat": "Observed improvements are not causal or guarantees of future performance.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args(argv)
    settings, rows = read_measurements(args.csv_path)
    if not rows:
        raise ValueError("CSV has no data records")
    baseline = dict(rows[0])
    result = repair(baseline, rows, settings)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
