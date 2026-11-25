"""Order-preserving, one-pass running summaries for configuration outcomes."""
from __future__ import annotations

import csv
import copy
import math
from pathlib import Path
from typing import Any, Iterable

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
DIRECTIONS = {"MRE-": "minimize", "ACC+": "maximize", "PRED40+": "maximize"}


def _empty(outcome: str) -> dict[str, Any]:
    return {"outcome": outcome, "direction": DIRECTIONS[outcome], "valid_count": 0,
            "mean": None, "sample_variance": None, "best_value": None,
            "best_configuration": None, "best_record_id": None}


def _snapshot(states: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in states.items()}


def running_summaries(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return immutable prefix snapshots; records are consumed exactly once."""
    states = {outcome: _empty(outcome) for outcome in OUTCOMES}
    # Internal Welford state is separate from the public snapshots.
    internal = {outcome: {"n": 0, "mean": 0.0, "m2": 0.0} for outcome in OUTCOMES}
    snapshots = []
    for position, record in enumerate(records):
        record_id = record.get("record_id", position)
        configuration = record.get("configuration")
        for outcome in OUTCOMES:
            try:
                value = float(record.get(outcome, ""))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            state, public = internal[outcome], states[outcome]
            state["n"] += 1
            delta = value - state["mean"]
            state["mean"] += delta / state["n"]
            state["m2"] += delta * (value - state["mean"])
            public["valid_count"] = state["n"]
            public["mean"] = state["mean"]
            public["sample_variance"] = state["m2"] / (state["n"] - 1) if state["n"] > 1 else None
            best = public["best_value"]
            better = best is None or (value < best if DIRECTIONS[outcome] == "minimize" else value > best)
            # Ties retain the first observed configuration and identifier.
            if better:
                public["best_value"] = value
                public["best_configuration"] = copy.deepcopy(configuration)
                public["best_record_id"] = record_id
        snapshots.append({"records_received": position + 1, "summaries": _snapshot(states)})
    return snapshots


def running_summaries_csv(path: str | Path) -> list[dict[str, Any]]:
    """Process CSV rows in file order; CSV order is a demonstration sequence."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        def mapped_records():
            for record_id, row in enumerate(csv.DictReader(handle), start=1):
                configuration = {field: row.get(field, "") for field in (
                    "N_estimators", "criterion", "Min_sample_leaves",
                    "Min_impurity_decrease", "Max_depth")}
                yield {"record_id": record_id, "configuration": configuration,
                       **{outcome: row.get(outcome, "") for outcome in OUTCOMES}}
        return running_summaries(mapped_records())
