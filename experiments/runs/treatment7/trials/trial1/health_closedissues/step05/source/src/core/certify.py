"""Step 3: certify whether an observed improvement is more than noise.

`same` follows the course's ezr procedure: two samples count as the same
only when the cohen, cliffs and ks tests all agree the gap is small.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from math import sqrt
from statistics import mean, stdev
from typing import Dict, List, Sequence, Union


def _oriented(values: Sequence[float], objective: str) -> List[float]:
    direction = -1.0 if objective.endswith("-") else 1.0
    return [direction * value for value in values]


def cohen(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Mean gap in pooled-standard-deviation units."""
    n, m = len(xs), len(ys)
    sx = stdev(xs) if n > 1 else 0.0
    sy = stdev(ys) if m > 1 else 0.0
    pooled = sqrt(((n - 1) * sx * sx + (m - 1) * sy * sy) / max(1, n + m - 2))
    return abs(mean(xs) - mean(ys)) / (pooled + 1e-32)


def cliffs(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank imbalance in 0..1: how often xs beat ys minus how often they lose."""
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return abs(gt - lt) / (len(xs) * len(ys))


def ks(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Largest gap between the two CDFs, in critical-value units."""
    xs, ys = sorted(xs), sorted(ys)
    n, m = len(xs), len(ys)
    gap = max(abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m) for v in set(xs) | set(ys))
    return gap / sqrt((n + m) / (n * m))


def same(xs: Sequence[float], ys: Sequence[float],
         cohen_limit: float = 0.35, cliffs_limit: float = 0.195, ks_limit: float = 1.36) -> bool:
    """True when the samples are statistically indistinguishable."""
    return cohen(xs, ys) <= cohen_limit and cliffs(xs, ys) <= cliffs_limit and ks(xs, ys) <= ks_limit


def certify(
    objective: str,
    treatment: Sequence[float],
    baseline: Sequence[float],
    alpha: float = 0.05,
    permutations: int = 2000,
    seed: int = 42,
) -> Dict[str, Union[float, bool]]:
    """Compare treatment and baseline with a permutation test plus `same`.

    A positive effect means the treatment is better after considering the
    objective's direction. A win needs a significant p-value AND `same`
    saying the samples differ.
    """

    if not treatment or not baseline:
        raise ValueError("treatment and baseline must both contain values")
    treatment_o = _oriented(treatment, objective)
    baseline_o = _oriented(baseline, objective)
    observed = mean(treatment_o) - mean(baseline_o)
    pooled = treatment_o + baseline_o
    size = len(treatment_o)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        rng.shuffle(pooled)
        if abs(mean(pooled[:size]) - mean(pooled[size:])) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (permutations + 1)
    indistinguishable = same(treatment_o, baseline_o)
    return {
        "effect": observed,
        "treatment_mean": mean(treatment_o),
        "baseline_mean": mean(baseline_o),
        "p_value": p_value,
        "significant": p_value < alpha,
        "cliffs": cliffs(treatment_o, baseline_o),
        "same": indistinguishable,
        "treatment_wins": observed > 0 and p_value < alpha and not indistinguishable,
    }
