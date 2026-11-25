#!/usr/bin/env python3
"""Sequential flow evaluator for Health-ClosedIssues HPO observations."""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from sklearn.utils.validation import check_array


OBJECTIVES = ("MRE-", "ACC+", "PRED40+")
CONFIGURATION = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
EXPECTED = OBJECTIVES + CONFIGURATION


@dataclass(frozen=True)
class Observation:
    ordinal: int
    values: Tuple[float, float, float]
    configuration: Tuple[object, ...]

    @property
    def utility(self) -> float:
        # All three source metrics are in [0, 1] in this dataset. Signs encode
        # direction: minimize MRE and maximize the two positive objectives.
        return -self.values[0] + self.values[1] + self.values[2]


class RunningEvaluator:
    """Consumes validated observations one at a time and keeps running state."""

    def __init__(self) -> None:
        self.seen = 0
        self.invalid = 0
        self.best: Optional[Observation] = None
        self.utility_sum = 0.0

    def observe(self, observation: Observation) -> bool:
        self.seen += 1
        self.utility_sum += observation.utility
        # Higher utility wins; exact ties keep the earliest observation.
        if self.best is None or observation.utility > self.best.utility:
            self.best = observation
            return True
        return False

    def metrics(self) -> Dict[str, object]:
        return {
            "observations": self.seen,
            "invalid_records": self.invalid,
            "mean_utility": self.utility_sum / self.seen if self.seen else None,
            "best_utility": self.best.utility if self.best else None,
            "best_ordinal": self.best.ordinal if self.best else None,
        }


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite value")
    return number


def parse_row(row: Dict[str, str], ordinal: int) -> Observation:
    if any(key not in row or row[key] is None or row[key].strip() == "" for key in EXPECTED):
        raise ValueError("missing required field")
    metrics = tuple(_finite(row[key]) for key in OBJECTIVES)
    if not all(0.0 <= value <= 1.0 for value in metrics):
        raise ValueError("objective outside [0, 1]")
    n_estimators = int(row["N_estimators"])
    leaves = int(row["Min_sample_leaves"])
    depth = int(row["Max_depth"])
    impurity = _finite(row["Min_impurity_decrease"])
    if n_estimators <= 0 or leaves <= 0 or depth <= 0 or impurity < 0:
        raise ValueError("invalid configuration value")
    # scikit-learn validates the numeric configuration vector as a final,
    # reproducible type/finite check without fitting a model.
    check_array([[n_estimators, leaves, impurity, depth]], dtype="numeric")
    config = (n_estimators, row["criterion"], leaves, impurity, depth)
    return Observation(ordinal, metrics, config)


def run(path: Path) -> Tuple[RunningEvaluator, list[str]]:
    evaluator = RunningEvaluator()
    events: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(EXPECTED):
            raise ValueError(f"unexpected header: {reader.fieldnames}")
        for ordinal, row in enumerate(reader, start=1):
            try:
                observation = parse_row(row, ordinal)
            except (ValueError, TypeError):
                evaluator.invalid += 1
                events.append(f"invalid ordinal={ordinal}")
                continue
            if evaluator.observe(observation):
                events.append(f"best ordinal={ordinal} utility={observation.utility:.12f}")
    return evaluator, events


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: flow_skill.py DATASET.csv", file=sys.stderr)
        return 2
    evaluator, events = run(Path(argv[1]))
    print("events=" + str(len(events)))
    print("running=" + str(evaluator.metrics()))
    if evaluator.best is not None:
        print("selected_configuration=" + repr(evaluator.best.configuration))
        print("selected_objectives=" + repr(evaluator.best.values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
