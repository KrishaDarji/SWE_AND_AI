#!/usr/bin/env python3
"""Streaming validation-result evaluator for model configurations."""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, TextIO


@dataclass(frozen=True)
class Observation:
    row_number: int
    values: Dict[str, str]
    mre: float
    acc: float
    pred40: float

    @property
    def score(self) -> float:
        # Normalize the three validation objectives to one higher-is-better score.
        return (1.0 - self.mre + self.acc + self.pred40) / 3.0


def read_observations(stream: TextIO) -> Iterable[Observation]:
    reader = csv.DictReader(stream)
    required = {"MRE-", "ACC+", "PRED40+"}
    if not required.issubset(reader.fieldnames or ()):
        raise ValueError("CSV must contain MRE-, ACC+, and PRED40+ columns")
    for row_number, row in enumerate(reader, 1):
        try:
            values = dict(row)
            yield Observation(row_number, values, float(row["MRE-"]),
                              float(row["ACC+"]), float(row["PRED40+"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid numeric validation result at row %d" % row_number) from exc


def dominates(left: Observation, right: Observation) -> bool:
    """Return whether left is at least as good in every objective and better in one."""
    left_values = (1.0 - left.mre, left.acc, left.pred40)
    right_values = (1.0 - right.mre, right.acc, right.pred40)
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


class FlowEvaluator:
    """Consume observations incrementally and retain only current summaries."""

    def __init__(self) -> None:
        self.count = 0
        self.best: Optional[Observation] = None
        self.frontier: List[Observation] = []

    def accept(self, observation: Observation) -> None:
        self.count += 1
        if self.best is None or observation.score > self.best.score:
            self.best = observation
        if any(dominates(existing, observation) for existing in self.frontier):
            return
        self.frontier = [existing for existing in self.frontier if not dominates(observation, existing)]
        self.frontier.append(observation)

    def summary(self) -> Dict[str, object]:
        if self.best is None:
            return {"observations": 0, "best": None, "pareto_size": 0}
        return {"observations": self.count, "best": {
            "row": self.best.row_number, "score": self.best.score,
            "MRE-": self.best.mre, "ACC+": self.best.acc, "PRED40+": self.best.pred40,
            "configuration": self.best.values,
        }, "pareto_size": len(self.frontier)}


def evaluate(stream: TextIO) -> Dict[str, object]:
    evaluator = FlowEvaluator()
    for observation in read_observations(stream):
        evaluator.accept(observation)
    return evaluator.summary()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="CSV file containing validation results")
    args = parser.parse_args(argv)
    try:
        with open(args.dataset, newline="", encoding="utf-8") as stream:
            result = evaluate(stream)
    except (OSError, ValueError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
