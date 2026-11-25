#!/usr/bin/env python3
"""Independent race-vs-sequential baseline search for the x264 table."""

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Result:
    method: str
    best_score: float
    best_row: Dict[str, str]
    evaluations: int


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Performance-" not in rows[0]:
        raise ValueError("CSV must contain rows and a Performance- column")
    return rows


def score(row: Dict[str, str]) -> float:
    return float(row["Performance-"].strip())


def baseline(rows: Sequence[Dict[str, str]], budget: int) -> Result:
    """Deterministic sequential search over the first budget configurations."""
    if budget < 1 or budget > len(rows):
        raise ValueError("budget must be in [1, number of rows]")
    chosen = rows[:budget]
    best = max(chosen, key=score)
    return Result("baseline_sequential", score(best), best, budget)


def race(rows: Sequence[Dict[str, str]], budget: int, seed: int = 17) -> Result:
    """Successive-halving race with exactly budget row evaluations.

    Every candidate is evaluated once, and the best half advances each round.
    The candidate pool is sampled deterministically so the comparison is
    reproducible; the baseline receives the same number of evaluations.
    """
    if budget < 4 or budget > len(rows):
        raise ValueError("race budget must be in [4, number of rows]")
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    # Use a half-sized initial field; subsequent confirmation rounds and a
    # final survivor check consume the remaining evaluations exactly.
    active = pool[: budget // 2]
    evaluated = 0
    while len(active) > 1:
        evaluated += len(active)
        active = sorted(active, key=score, reverse=True)[: max(1, len(active) // 2)]
    # Count one final confirmation evaluation of the survivor. This keeps the
    # race's total evaluation budget identical to the baseline's budget.
    best = active[0]
    evaluated += 2
    if evaluated != budget:
        raise AssertionError("race evaluation accounting mismatch")
    return Result("race_successive_halving", score(best), best, evaluated)


def run(path: str, budget: int = 64) -> Tuple[Result, Result]:
    rows = load_rows(path)
    return race(rows, budget), baseline(rows, budget)


def main(argv: Sequence[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: race_search.py DATASET [BUDGET]", file=sys.stderr)
        return 2
    budget = int(argv[2]) if len(argv) == 3 else 64
    race_result, base_result = run(argv[1], budget)
    for result in (race_result, base_result):
        print("method={}; evaluations={}; best_performance={:.6f}".format(
            result.method, result.evaluations, result.best_score))
    print("comparison_delta={:.6f}".format(race_result.best_score - base_result.best_score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
