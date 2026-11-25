"""Budget-limited configuration selection for lower-is-better Performance-."""
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Result:
    evaluation_order: tuple
    observations: tuple
    budget_used: int
    recommended_id: str
    recommended_configuration: tuple


class RunningNumeric:
    """EZR-inspired one-pass numeric summary (mean and sample deviation)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def deviation(self):
        return 0.0 if self.n < 2 else math.sqrt(max(self.m2, 0.0) / (self.n - 1))


def _key(configuration):
    try:
        return tuple(configuration)
    except TypeError as exc:
        raise ValueError("each candidate must be a finite sequence") from exc


def select(candidates, evaluate, budget, seed=42):
    """Evaluate at most budget unique candidates and freeze the best result.

    `evaluate` is the only source of outcomes; candidates contain settings only.
    Identifier order is the input order, represented as c00000, c00001, ... .
    Ties prefer the lower identifier (earlier input position).
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("candidates must not be empty")
    normalized = [_key(c) for c in candidates]
    seen = set()
    unique = []
    for index, config in enumerate(normalized):
        if config not in seen:
            seen.add(config)
            unique.append((index, config))
    rng = random.Random(seed)
    order = list(unique)
    rng.shuffle(order)
    chosen = order[:min(budget, len(order))]
    observations = []
    summary = RunningNumeric()
    for index, config in chosen:
        value = evaluate(config)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("evaluation result must be a finite number")
        identifier = f"c{index:05d}"
        observations.append((identifier, config, float(value)))
        summary.add(float(value))
    best = min(observations, key=lambda row: (row[2], int(row[0][1:])))
    return Result(tuple(row[0] for row in observations), tuple(observations),
                  len(observations), best[0], best[1])


def audit(candidates, outcome_lookup):
    """Retrospective full-data minimum; never called by select."""
    rows = [(f"c{i:05d}", _key(c), float(outcome_lookup(_key(c))))
            for i, c in enumerate(candidates)]
    if not rows:
        raise ValueError("candidates must not be empty")
    if any(not math.isfinite(row[2]) for row in rows):
        raise ValueError("audit outcomes must be finite")
    return min(rows, key=lambda row: (row[2], int(row[0][1:])))
