"""Compare the existing search with a fixed-budget random baseline."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    strategy: str
    configuration: dict
    objective: float
    evaluations: int


@dataclass(frozen=True)
class RaceResult:
    objective_measure: str
    budget: int
    system: SearchResult
    baseline: SearchResult


def _configurations(path, measure):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        if measure not in reader.fieldnames:
            raise ValueError(f"unknown objective measure column: {measure}")
        settings = [name for name in reader.fieldnames if name != measure]
        groups = {}
        for row_number, row in enumerate(reader, 2):
            try:
                value = float(str(row[measure]).strip())
            except ValueError as exc:
                raise ValueError(f"row {row_number}: measure is not numeric") from exc
            key = tuple(row[name].strip() for name in settings)
            groups.setdefault(key, []).append(value)
    return settings, [(key, sum(values) / len(values)) for key, values in groups.items()]


def _best(settings, rows, indexes, strategy):
    key, objective = min((rows[i] for i in indexes), key=lambda item: (item[1], item[0]))
    return SearchResult(strategy, dict(zip(settings, key)), objective, len(indexes))


def compare_searches(path: str, measure: str = "", budget: int = 64,
                     seed: int = 0) -> RaceResult:
    """Race row-order system search against seeded random search at equal budget."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if not fieldnames:
        raise ValueError("CSV must contain a header")
    target = measure or fieldnames[-1]
    settings, rows = _configurations(path, target)
    if not rows:
        raise ValueError("CSV must contain at least one data row")
    budget = min(budget, len(rows))
    system_indexes = list(range(budget))
    baseline_indexes = random.Random(seed).sample(range(len(rows)), budget)
    return RaceResult(target, budget, _best(settings, rows, system_indexes, "system"),
                      _best(settings, rows, baseline_indexes, "baseline"))
