#!/usr/bin/env python3
"""Sequential proposal assessment for NASA93DEM (Treatment 3, Trial 2)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, TextIO


REQUIRED = ("idX", "Effort-", "Defects-", "MonthsX")
# Fixed planning scales make scores comparable as records arrive.
SCALES = {"Effort-": 1000.0, "Defects-": 10000.0, "MonthsX": 100.0}


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    effort: float
    defects: float
    months: float
    score: float


@dataclass
class EvaluationState:
    seen_count: int = 0
    valid_count: int = 0
    error_count: int = 0
    best: Proposal | None = None
    running_scores: list[float] | None = None

    def __post_init__(self) -> None:
        if self.running_scores is None:
            self.running_scores = []


def validate_header(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("dataset has no header")
    missing = [name for name in REQUIRED if name not in fieldnames]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))


def parse_proposal(row: dict[str, str]) -> Proposal:
    proposal_id = (row.get("idX") or "").strip()
    if not proposal_id:
        raise ValueError("missing idX")
    values: dict[str, float] = {}
    for name in REQUIRED[1:]:
        raw = (row.get(name) or "").strip()
        if not raw:
            raise ValueError(f"missing {name}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"non-numeric {name}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name}")
        values[name] = value
    score = sum(values[name] / SCALES[name] for name in values)
    return Proposal(proposal_id, values["Effort-"], values["Defects-"], values["MonthsX"], score)


def consider(state: EvaluationState, proposal: Proposal) -> None:
    state.valid_count += 1
    state.running_scores.append(proposal.score)
    # Lower cost wins; ID is the deterministic tie-breaker.
    if state.best is None or (proposal.score, proposal.proposal_id) < (state.best.score, state.best.proposal_id):
        state.best = proposal


def evaluate(stream: TextIO) -> tuple[EvaluationState, list[str]]:
    reader = csv.DictReader(stream)
    validate_header(reader.fieldnames)
    state = EvaluationState()
    errors: list[str] = []
    for line_number, row in enumerate(reader, start=2):
        state.seen_count += 1
        try:
            consider(state, parse_proposal(row))
        except ValueError as exc:
            state.error_count += 1
            errors.append(f"line {line_number}: {exc}")
    return state, errors


def run(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as stream:
        state, errors = evaluate(stream)
    if state.seen_count == 0:
        raise ValueError("dataset contains no records")
    return {"records_seen": state.seen_count, "valid_records": state.valid_count,
            "error_count": state.error_count, "best": asdict(state.best) if state.best else None,
            "running_scores": state.running_scores, "errors": errors}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.dataset), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
