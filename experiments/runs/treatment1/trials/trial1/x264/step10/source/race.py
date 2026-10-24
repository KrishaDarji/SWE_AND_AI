#!/usr/bin/env python3
"""Race candidate configurations against a fixed-budget random-search baseline."""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    values: dict[str, str]
    score: float


def load_records(path: str | Path, objective: str = "Performance-") -> list[Record]:
    with open(path, newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or objective not in rows.fieldnames:
            raise ValueError(f"CSV must contain objective column {objective!r}")
        records = []
        for row in rows:
            try:
                records.append(Record(dict(row), float(row[objective].strip())))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid objective value in row {row!r}") from exc
    if not records:
        raise ValueError("CSV contains no data rows")
    return records


def _best(records: Iterable[Record], maximize: bool) -> Record:
    return max(records, key=lambda r: r.score) if maximize else min(records, key=lambda r: r.score)


def random_search(records: list[Record], budget: int, seed: int, maximize: bool = False) -> tuple[Record, int]:
    """Fixed-budget baseline; sampling is without replacement."""
    if budget < 1:
        raise ValueError("budget must be positive")
    sample = random.Random(seed).sample(records, min(budget, len(records)))
    return _best(sample, maximize), len(sample)


def race_search(records: list[Record], budget: int, rounds: int = 3, seed: int = 7,
                maximize: bool = False) -> tuple[Record, int]:
    """Successive-halving race: score all arms in rounds and retain the better half."""
    if budget < 1 or rounds < 1:
        raise ValueError("budget and rounds must be positive")
    arms = random.Random(seed).sample(records, min(budget, len(records)))
    evaluated: list[Record] = []
    for round_no in range(rounds):
        evaluated.extend(arms)
        if len(arms) == 1:
            break
        keep = max(1, (len(arms) + 1) // 2)
        # A deterministic tie-break by original row values avoids unstable races.
        arms = sorted(arms, key=lambda r: (r.score, tuple(r.values.items())), reverse=maximize)[:keep]
    return _best(evaluated, maximize), len(evaluated)


def compare(path: str | Path, budget: int = 32, rounds: int = 3, seed: int = 7,
            objective: str = "Performance-", maximize: bool = False) -> dict[str, object]:
    records = load_records(path, objective)
    baseline, baseline_n = random_search(records, budget, seed, maximize)
    raced, raced_n = race_search(records, budget, rounds, seed, maximize)
    return {"rows": len(records), "objective": objective, "maximize": maximize,
            "baseline_score": baseline.score, "race_score": raced.score,
            "baseline_evaluations": baseline_n, "race_evaluations": raced_n,
            "race_improvement": raced.score - baseline.score if maximize else baseline.score - raced.score,
            "baseline_config": baseline.values, "race_config": raced.values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maximize", action="store_true")
    args = parser.parse_args()
    result = compare(args.csv_path, args.budget, args.rounds, args.seed, maximize=args.maximize)
    for key, value in result.items():
        if key.endswith("_config"):
            continue
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
