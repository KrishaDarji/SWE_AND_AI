#!/usr/bin/env python3
"""Retrospective selection of an observed configuration under an estimator cap."""
import argparse
import csv
import math
import sys

CONFIGURATION = ("N_estimators", "criterion", "Min_sample_leaves",
                 "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")
REQUIRED = CONFIGURATION + OUTCOMES


class SelectionError(ValueError):
    """An invalid cap or input record set."""


def finite_number(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SelectionError("{} must be numeric".format(field))
    if not math.isfinite(number):
        raise SelectionError("{} must be finite".format(field))
    return number


def parse_cap(value):
    cap = finite_number(value, "cap")
    if cap < 0:
        raise SelectionError("cap must be nonnegative")
    if not cap.is_integer():
        raise SelectionError("cap must be an integer")
    return int(cap)


def read_records(stream):
    reader = csv.DictReader(stream)
    if not reader.fieldnames or any(name not in reader.fieldnames for name in REQUIRED):
        missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
        raise SelectionError("missing required columns: " + ", ".join(missing))
    records = []
    number = 0
    for number, row in enumerate(reader, 1):
        record = {"record_id": number}
        valid = True
        for field in CONFIGURATION:
            if field == "criterion":
                criterion = (row.get(field) or "").strip()
                valid = valid and bool(criterion) and criterion != "?"
            else:
                try:
                    record[field] = finite_number(row.get(field), field)
                except SelectionError:
                    valid = False
        for field in OUTCOMES:
            try:
                record[field] = finite_number(row.get(field), field)
            except SelectionError:
                valid = False
        if valid:
            record["criterion"] = row["criterion"].strip()
            records.append(record)
    return records, number


def choose(stream, cap):
    cap = parse_cap(cap)
    records, raw_total = read_records(stream)
    feasible = [r for r in records if r["N_estimators"] <= cap]
    if not feasible:
        raise SelectionError("no feasible records (cap={}, valid={})".format(cap, len(records)))
    best = min(feasible, key=lambda r: (r["MRE-"], -r["ACC+"], -r["PRED40+"], r["record_id"]))
    return {"cap": cap, "feasible_count": len(feasible), "valid_count": len(records),
            "total_count": raw_total,
            "record_id": best["record_id"],
            "configuration": {field: best[field] for field in CONFIGURATION},
            "outcomes": {field: best[field] for field in OUTCOMES}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Choose the best observed configuration under an N_estimators cap.")
    parser.add_argument("csv_path")
    parser.add_argument("--cap", default="100")
    args = parser.parse_args(argv)
    try:
        with open(args.csv_path, newline="", encoding="utf-8") as source:
            result = choose(source, args.cap)
    except (OSError, csv.Error, SelectionError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print("cap={cap} feasible={feasible_count} valid={valid_count} total={total_count} record_id={record_id}".format(**result))
    print("configuration=" + repr(result["configuration"]))
    print("outcomes=" + repr(result["outcomes"]))
    print("selection uses observed outcomes; it does not predict unseen settings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
