"""Incremental, order-preserving summaries for configuration outcomes."""
import csv
import math
from copy import deepcopy
from pathlib import Path

CONFIGURATION_FIELDS = ["N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth"]
OUTCOME_FIELDS = ["MRE-", "ACC+", "PRED40+"]
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _configuration(row):
    return {field: row.get(field) for field in CONFIGURATION_FIELDS}


def _empty(field):
    return {"outcome": field, "direction": DIRECTIONS[field], "valid_count": 0,
            "mean": None, "sample_variance": None, "best_value": None,
            "best_configuration": None, "best_record_id": None}


def _update(summary, field, value, configuration, record_id):
    state = summary[field]
    if value is None:
        return
    count = state["valid_count"] + 1
    old_mean = state["mean"]
    delta = value if old_mean is None else value - old_mean
    mean = value if old_mean is None else old_mean + delta / count
    old_m2 = 0.0 if state["sample_variance"] is None else state["sample_variance"] * (count - 2)
    m2 = old_m2 + (value - mean) * delta
    state["valid_count"] = count
    state["mean"] = mean
    state["sample_variance"] = m2 / (count - 1) if count > 1 else None
    best = state["best_value"]
    improves = best is None or (value < best if DIRECTIONS[field] == "minimize" else value > best)
    if improves:
        state["best_value"] = value
        state["best_configuration"] = deepcopy(configuration)
        state["best_record_id"] = record_id


def snapshots(records):
    """Yield defensive snapshots after each record from any iterable of dictionaries."""
    summary = {field: _empty(field) for field in OUTCOME_FIELDS}
    for index, row in enumerate(records):
        record_id = row.get("record_id", index)
        configuration = _configuration(row)
        for field in OUTCOME_FIELDS:
            _update(summary, field, _finite(row.get(field)), configuration, record_id)
        yield deepcopy({"record_id": record_id, "outcomes": summary})


def flow_csv(path):
    """Consume CSV rows incrementally; CSV order is a demonstration sequence."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        yield from snapshots(csv.DictReader(handle))


def final_summary(path):
    result = None
    for result in flow_csv(path):
        pass
    return {field: _empty(field) for field in OUTCOME_FIELDS} if result is None else result["outcomes"]


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    args = parser.parse_args()
    print(json.dumps(final_summary(args.dataset), indent=2, sort_keys=True))
