"""Evidence-based comparison of the costs of two project groups."""

from dataclasses import dataclass
from math import floor
from random import Random
from statistics import mean, median
from typing import Iterable

from .model import Project


@dataclass(frozen=True)
class CostComparison:
    """Result where a negative difference means process A costs less."""

    process_a: str
    process_b: str
    cost_field: str
    sample_a: int
    sample_b: int
    median_a: float
    median_b: float
    mean_a: float
    mean_b: float
    difference_mean_a_minus_b: float
    confidence: float
    difference_interval: tuple
    permutation_p_value: float
    genuinely_less_costly: bool
    method: str

    def to_dict(self):
        return self.__dict__.copy()


def _percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = floor(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def certify_less_costly(projects_a: Iterable[Project], projects_b: Iterable[Project],
                        process_a="A", process_b="B", cost_field="effort",
                        confidence=.95, resamples=2000, seed=1) -> CostComparison:
    """Certify A as less costly only with a one-sided bootstrap and permutation test."""
    a, b = list(projects_a), list(projects_b)
    if not a or not b:
        raise ValueError("both process groups must be non-empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    try:
        av = [float(getattr(p, cost_field)) for p in a]
        bv = [float(getattr(p, cost_field)) for p in b]
    except AttributeError as exc:
        raise ValueError("unknown cost field: %s" % cost_field) from exc
    observed = mean(av) - mean(bv)
    rng = Random(seed)
    bootstrap = []
    for _ in range(resamples):
        bootstrap.append(mean(rng.choice(av) for _ in av) - mean(rng.choice(bv) for _ in bv))
    alpha = 1 - confidence
    interval = (_percentile(bootstrap, alpha / 2), _percentile(bootstrap, 1 - alpha / 2))
    pooled = av + bv
    extreme = 0
    for _ in range(resamples):
        rng.shuffle(pooled)
        difference = mean(pooled[:len(av)]) - mean(pooled[len(av):])
        if difference <= observed:
            extreme += 1
    p_value = (extreme + 1) / (resamples + 1)
    return CostComparison(process_a, process_b, cost_field, len(av), len(bv),
                           median(av), median(bv), mean(av), mean(bv), observed,
                           confidence, interval, p_value,
                           interval[1] < 0 and p_value < alpha,
                           "bootstrap confidence interval plus one-sided permutation test")
