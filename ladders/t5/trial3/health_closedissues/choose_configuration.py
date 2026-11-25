#!/usr/bin/env python3
"""Choose the best observed configuration under an estimator cap."""
import csv
import math
import sys

CONFIGURATION_FIELDS = (
    "N_estimators", "criterion", "Min_sample_leaves",
    "Min_impurity_decrease", "Max_depth",
)
OUTCOME_FIELDS = ("MRE-", "ACC+", "PRED40+")


class SelectionError(ValueError):
    """Raised when the input or cap cannot support a selection."""


def _finite_number(value, field, record_id):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SelectionError("record {} has nonnumeric {}".format(record_id, field))
    if not math.isfinite(number):
        raise SelectionError("record {} has nonfinite {}".format(record_id, field))
    return number


def _record_is_eligible(row, record_id, cap):
    for field in CONFIGURATION_FIELDS:
        if field not in row or row[field] is None or not str(row[field]).strip():
            return False
    if not str(row["criterion"]).strip() or str(row["criterion"]).strip() == "?":
        return False
    for field in OUTCOME_FIELDS:
        if field not in row or row[field] is None or not str(row[field]).strip():
            return False
        _finite_number(row[field], field, record_id)
    estimators = _finite_number(row["N_estimators"], "N_estimators", record_id)
    _finite_number(row["Min_sample_leaves"], "Min_sample_leaves", record_id)
    _finite_number(row["Min_impurity_decrease"], "Min_impurity_decrease", record_id)
    _finite_number(row["Max_depth"], "Max_depth", record_id)
    return estimators <= cap


def choose_configuration(csv_path, cap):
    """Return the selected observed row and counts under a finite cap."""
    if isinstance(cap, bool):
        raise SelectionError("cap must be a finite nonnegative number")
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        raise SelectionError("cap must be a finite nonnegative number")
    if not math.isfinite(cap) or cap < 0:
        raise SelectionError("cap must be a finite nonnegative number")
    records = []
    total_count = 0
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(field not in reader.fieldnames
                                        for field in CONFIGURATION_FIELDS + OUTCOME_FIELDS):
            raise SelectionError("input is missing required columns")
        for record_id, row in enumerate(reader, start=1):
            if not any(value is not None and str(value).strip() for value in row.values()):
                continue
            total_count += 1
            if _record_is_eligible(row, record_id, cap):
                row = dict(row)
                row["record_id"] = record_id
                records.append(row)
    if not records:
        raise SelectionError("no feasible records")
    selected = min(records, key=lambda row: (
        _finite_number(row["MRE-"], "MRE-", row["record_id"]),
        -_finite_number(row["ACC+"], "ACC+", row["record_id"]),
        -_finite_number(row["PRED40+"], "PRED40+", row["record_id"]),
        row["record_id"],
    ))
    return {"cap": cap, "feasible_count": len(records), "total_count": total_count, "selected": selected}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: choose_configuration.py CSV_PATH CAP", file=sys.stderr)
        return 2
    try:
        result = choose_configuration(argv[0], argv[1])
    except (OSError, SelectionError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    selected = result["selected"]
    print("cap={}".format(result["cap"]))
    print("feasible_count={} total_count={}".format(result["feasible_count"], result["total_count"]))
    print("record_id={}".format(selected["record_id"]))
    print("configuration=" + ",".join("{}={}".format(field, selected[field]) for field in CONFIGURATION_FIELDS))
    print("outcomes=" + ",".join("{}={}".format(field, selected[field]) for field in OUTCOME_FIELDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
