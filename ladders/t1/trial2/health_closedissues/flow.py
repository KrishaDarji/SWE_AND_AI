"""Evaluate model configurations as validation results arrive one at a time."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ValidationResult:
    """One configuration and its validation metrics."""

    configuration: Mapping[str, Any]
    metrics: Mapping[str, float]


class FlowEvaluator:
    """Keep the best configuration without requiring all results in memory."""

    def __init__(self, directions: Mapping[str, str]):
        if not directions:
            raise ValueError("at least one metric direction is required")
        if any(value not in ("min", "max") for value in directions.values()):
            raise ValueError("directions must be 'min' or 'max'")
        self.directions = dict(directions)
        self.seen = 0
        self.invalid = 0
        self._best: Optional[ValidationResult] = None
        self._best_key: Optional[Tuple[float, ...]] = None

    def observe(self, result: ValidationResult) -> bool:
        """Consume exactly one result; return True when the incumbent changes."""
        try:
            values = tuple(float(result.metrics[name]) for name in self.directions)
        except (KeyError, TypeError, ValueError):
            self.invalid += 1
            return False
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            self.invalid += 1
            return False
        self.seen += 1
        key = tuple(value if self.directions[name] == "min" else -value
                    for name, value in zip(self.directions, values))
        changed = self._best_key is None or key < self._best_key
        if changed:
            self._best_key = key
            self._best = result
        return changed

    @property
    def best(self) -> Optional[ValidationResult]:
        return self._best


def stream_csv(path: Path, evaluator: FlowEvaluator) -> int:
    """Read and evaluate a CSV row-by-row, returning processed row count."""
    processed = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metrics = {name: float(row[name]) for name in evaluator.directions}
            configuration = {key: value for key, value in row.items()
                             if key not in evaluator.directions}
            evaluator.observe(ValidationResult(configuration, metrics))
            processed += 1
    return processed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args(argv)
    evaluator = FlowEvaluator({"MRE-": "min", "ACC+": "max", "PRED40+": "max"})
    rows = stream_csv(args.csv_path, evaluator)
    best = evaluator.best
    print(json.dumps({"rows": rows, "valid": evaluator.seen, "invalid": evaluator.invalid,
                      "best": None if best is None else dict(best.configuration),
                      "metrics": None if best is None else dict(best.metrics)}, sort_keys=True))
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
