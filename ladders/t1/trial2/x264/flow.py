#!/usr/bin/env python3
"""Evaluate benchmark configurations incrementally as results arrive."""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TextIO


@dataclass(frozen=True)
class Observation:
    index: int
    configuration: Dict[str, str]
    performance: float


class FlowEvaluator:
    """Keep the best (lowest-performance) observation seen so far."""

    def __init__(self, objective_column: str = "Performance-") -> None:
        self.objective_column = objective_column
        self.observations = 0
        self.best: Optional[Observation] = None

    def accept(self, row: Dict[str, str]) -> Dict[str, object]:
        if self.objective_column not in row:
            raise ValueError("missing objective column: " + self.objective_column)
        raw = row[self.objective_column].strip()
        try:
            performance = float(raw)
        except ValueError as exc:
            raise ValueError("invalid performance value: " + raw) from exc
        if performance != performance:
            raise ValueError("performance must not be NaN")
        configuration = {key: value for key, value in row.items() if key != self.objective_column}
        self.observations += 1
        observation = Observation(self.observations, configuration, performance)
        improved = self.best is None or performance < self.best.performance
        if improved:
            self.best = observation
        assert self.best is not None
        return {
            "index": observation.index,
            "performance": performance,
            "improved": improved,
            "best_performance": self.best.performance,
            "best_index": self.best.index,
            "best_configuration": self.best.configuration,
        }


def evaluate(stream: TextIO, evaluator: Optional[FlowEvaluator] = None) -> Iterable[Dict[str, object]]:
    evaluator = evaluator or FlowEvaluator()
    reader = csv.DictReader(stream)
    if not reader.fieldnames or "Performance-" not in reader.fieldnames:
        raise ValueError("CSV must contain a Performance- column")
    for row in reader:
        if not row or all(value is None or not value.strip() for value in row.values()):
            continue
        yield evaluator.accept(row)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", help="CSV path; stdin when omitted")
    args = parser.parse_args(argv)
    try:
        if args.csv_path:
            with open(args.csv_path, "r", newline="", encoding="utf-8") as stream:
                results = evaluate(stream)
                for result in results:
                    print(json.dumps(result, sort_keys=True))
        else:
            for result in evaluate(sys.stdin):
                print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
