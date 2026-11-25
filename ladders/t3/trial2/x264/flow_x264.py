#!/usr/bin/env python3
"""Independent sequential flow evaluator for the x264 benchmark dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

EXPECTED_OBJECTIVE = "Performance-"
CONFIG_WIDTH = 16


@dataclass(frozen=True)
class Observation:
    configuration: Tuple[int, ...]
    objective: float
    row_number: int


@dataclass
class EvaluationState:
    observations_seen: int = 0
    valid_observations: int = 0
    invalid_records: int = 0
    best: Optional[Observation] = None
    objective_sum: float = 0.0
    objective_min: Optional[float] = None
    objective_max: Optional[float] = None
    _trace: list[str] = field(default_factory=list)

    def ingest(self, record: Sequence[str], row_number: int) -> None:
        self.observations_seen += 1
        observation = parse_record(record, row_number)
        if observation is None:
            self.invalid_records += 1
            return
        self.valid_observations += 1
        self.objective_sum += observation.objective
        self.objective_min = observation.objective if self.objective_min is None else min(self.objective_min, observation.objective)
        self.objective_max = observation.objective if self.objective_max is None else max(self.objective_max, observation.objective)
        self._trace.append(f"{row_number}:{observation.objective:.12g}")
        # Strictly smaller wins; equal values retain the earliest row.
        if self.best is None or observation.objective < self.best.objective:
            self.best = observation

    def digest(self) -> str:
        return hashlib.sha256("|".join(self._trace).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {
            "observations_seen": self.observations_seen,
            "valid_observations": self.valid_observations,
            "invalid_records": self.invalid_records,
            "best_row": None if self.best is None else self.best.row_number,
            "best_objective": None if self.best is None else self.best.objective,
            "best_configuration": None if self.best is None else list(self.best.configuration),
            "mean_objective": None if not self.valid_observations else self.objective_sum / self.valid_observations,
            "objective_min": self.objective_min,
            "objective_max": self.objective_max,
            "state_digest": self.digest(),
        }


def parse_record(record: Sequence[str], row_number: int) -> Optional[Observation]:
    if len(record) != CONFIG_WIDTH + 1:
        return None
    try:
        configuration = tuple(int(value) for value in record[:CONFIG_WIDTH])
        if any(value not in (0, 1) for value in configuration):
            return None
        objective = float(record[-1])
        if not math.isfinite(objective):
            return None
    except (TypeError, ValueError):
        return None
    return Observation(configuration, objective, row_number)


def read_records(path: Path) -> Iterator[tuple[int, list[str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("dataset is empty") from exc
        if len(header) != CONFIG_WIDTH + 1 or header[-1] != EXPECTED_OBJECTIVE:
            raise ValueError(f"unexpected header: {header!r}")
        if header[:CONFIG_WIDTH] != [f for f in header[:CONFIG_WIDTH]]:
            raise ValueError("invalid configuration header")
        for row_number, record in enumerate(reader, start=2):
            yield row_number, record


def evaluate(path: Path) -> EvaluationState:
    state = EvaluationState()
    for row_number, record in read_records(path):
        state.ingest(record, row_number)
    return state


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    state = evaluate(args.dataset)
    elapsed = time.perf_counter() - started
    print(json.dumps({"dataset": str(args.dataset), "runtime_seconds": elapsed, **state.summary()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
