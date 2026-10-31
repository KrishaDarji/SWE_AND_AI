"""Budgeted configuration selection for lower-is-better Performance-."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Selection:
    evaluations: tuple
    observations: tuple
    recommended_id: str | None
    recommended_config: tuple | None
    budget_used: int


def _key(config):
    try:
        return tuple(config)
    except TypeError as exc:
        raise ValueError("configuration must be iterable") from exc


def select_configurations(candidates, budget, evaluate, seed=42):
    """Evaluate a deterministic random permutation, then return its minimum.

    The callback is the only source of outcomes. IDs are eval-001, eval-002,
    ... in evaluation order, so ties resolve in favor of the earlier record.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    unique = []
    seen = set()
    for candidate in candidates:
        config = _key(candidate)
        if config not in seen:
            seen.add(config)
            unique.append(config)
    if not unique:
        raise ValueError("candidates must not be empty")
    order = list(range(len(unique)))
    random.Random(seed).shuffle(order)
    chosen = order[:min(budget, len(order))]
    records = []
    for position, index in enumerate(chosen, 1):
        value = evaluate(unique[index])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("evaluation result must be numeric and finite")
        if not math.isfinite(value):
            raise ValueError("evaluation result must be numeric and finite")
        records.append((f"eval-{position:03d}", unique[index], float(value)))
    best = min(records, key=lambda record: (record[2], int(record[0][5:])))
    return Selection(tuple(r[0] for r in records), tuple(records), best[0], best[1], len(records))


def read_dataset(path):
    import csv
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    names = [name for name in rows[0] if name != "Performance-"]
    return names, [tuple(int(row[name]) for name in names) for row in rows], [float(row["Performance-"]) for row in rows]
