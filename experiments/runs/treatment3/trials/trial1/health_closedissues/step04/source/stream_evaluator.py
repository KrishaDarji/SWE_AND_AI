"""One-at-a-time evaluation of validation results for HPO configurations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OBJECTIVES = ("MRE-", "ACC+", "PRED40+")
CONFIG_COLUMNS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")


@dataclass(frozen=True)
class Result:
    outcomes: Dict[str, float]
    configuration: Dict[str, object]
    index: int


class StreamEvaluator:
    """Update ranking state after each validation result; no future rows are used."""

    def __init__(self) -> None:
        self.results: List[Result] = []
        self.best: Optional[Result] = None
        self.pareto: List[Result] = []
        self.best_score: Optional[float] = None

    @staticmethod
    def _score(result: Result, seen: Sequence[Result]) -> float:
        # Min-max normalization is based only on rows already seen. Lower is better.
        values = {k: [r.outcomes[k] for r in seen] for k in OBJECTIVES}
        total = 0.0
        for key in OBJECTIVES:
            lo, hi = min(values[key]), max(values[key])
            normalized = 0.0 if hi == lo else (result.outcomes[key] - lo) / (hi - lo)
            total += normalized if key == "MRE-" else 1.0 - normalized
        return total / len(OBJECTIVES)

    @staticmethod
    def _dominates(a: Result, b: Result) -> bool:
        no_worse = a.outcomes["MRE-"] <= b.outcomes["MRE-"] and a.outcomes["ACC+"] >= b.outcomes["ACC+"] and a.outcomes["PRED40+"] >= b.outcomes["PRED40+"]
        strictly = any((a.outcomes[k] < b.outcomes[k]) if k == "MRE-" else (a.outcomes[k] > b.outcomes[k]) for k in OBJECTIVES)
        return no_worse and strictly

    def add(self, row: Dict[str, str]) -> Result:
        outcomes = {k: float(row[k]) for k in OBJECTIVES}
        if not all(math.isfinite(v) for v in outcomes.values()):
            raise ValueError("outcomes must be finite numbers")
        configuration = {k: row[k] for k in CONFIG_COLUMNS}
        result = Result(outcomes, configuration, len(self.results) + 1)
        self.results.append(result)
        ranges = {k: (min(r.outcomes[k] for r in self.results), max(r.outcomes[k] for r in self.results)) for k in OBJECTIVES}
        if not any(self._dominates(existing, result) for existing in self.pareto):
            self.pareto = [existing for existing in self.pareto if not self._dominates(result, existing)]
            self.pareto.append(result)
        # The new arrival is scored against all observations available now.
        # Existing decisions are not retroactively changed by future rows.
        score = sum((result.outcomes[k] - ranges[k][0]) / (ranges[k][1] - ranges[k][0]) if ranges[k][1] != ranges[k][0] else 0.0 if k == "MRE-" else 1.0 for k in OBJECTIVES) / len(OBJECTIVES)
        if self.best is None or score <= self.best_score:
            self.best = result
            self.best_score = score
        return result


def read_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(OBJECTIVES + CONFIG_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        yield from reader


def run(path: Path, limit: Optional[int] = None) -> Dict[str, object]:
    evaluator = StreamEvaluator()
    for row_number, row in enumerate(read_rows(path), 1):
        evaluator.add(row)
        if limit is not None and row_number >= limit:
            break
    if evaluator.best is None:
        raise ValueError("no validation results")
    return {"seen": len(evaluator.results), "best": evaluator.best.__dict__, "pareto_size": len(evaluator.pareto)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.csv_path, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
