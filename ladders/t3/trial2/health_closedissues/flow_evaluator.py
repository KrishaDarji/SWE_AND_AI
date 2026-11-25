"""Sequential, deterministic evaluation of Health-ClosedIssues observations."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

REQUIRED = ("MRE-", "ACC+", "PRED40+", "N_estimators", "criterion",
            "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
METRICS = ("MRE-", "ACC+", "PRED40+")


@dataclass
class EvaluationState:
    observations: int = 0
    valid: int = 0
    invalid: int = 0
    best: dict | None = None
    running: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in METRICS})
    _totals: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in METRICS})

    def _key(self, row: Mapping[str, str]) -> tuple:
        # Lower MRE is better; ties prefer larger positive metrics, then stable config order.
        return (float(row["MRE-"]), -float(row["ACC+"]), -float(row["PRED40+"]),
                int(row["N_estimators"]), row["criterion"],
                int(row["Min_sample_leaves"]), float(row["Min_impurity_decrease"]),
                int(row["Max_depth"]))

    def observe(self, row: Mapping[str, str]) -> bool:
        """Consume exactly one record. Invalid records are counted and ignored."""
        self.observations += 1
        try:
            if any(k not in row or str(row[k]).strip() == "" for k in REQUIRED):
                raise ValueError("missing field")
            values = {m: float(row[m]) for m in METRICS}
            ints = [int(row[k]) for k in ("N_estimators", "Min_sample_leaves", "Max_depth")]
            if any(not math.isfinite(v) for v in values.values()) or any(v < 0 for v in values.values()):
                raise ValueError("invalid metric")
            if any(v < 0 for v in ints) or not math.isfinite(float(row["Min_impurity_decrease"])):
                raise ValueError("invalid configuration")
            candidate = dict(row)
            candidate["_key"] = self._key(row)
        except (ValueError, TypeError, OverflowError):
            self.invalid += 1
            return False
        self.valid += 1
        for metric in METRICS:
            self._totals[metric] += values[metric]
            self.running[metric] = self._totals[metric] / self.valid
        if self.best is None or candidate["_key"] < self.best["_key"]:
            self.best = candidate
        return True


def evaluate_rows(rows: Iterable[Mapping[str, str]]) -> EvaluationState:
    state = EvaluationState()
    for row in rows:
        state.observe(row)
    return state


def evaluate_csv(path: str | Path) -> EvaluationState:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED:
            raise ValueError(f"expected header {REQUIRED}, got {reader.fieldnames}")
        return evaluate_rows(reader)

