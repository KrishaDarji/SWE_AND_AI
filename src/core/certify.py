"""Step 3: certify whether an observed improvement is more than noise."""

from __future__ import annotations

import random
from statistics import mean
from typing import Dict, List, Sequence, Union


def _better_is_higher(objective: str) -> bool:
    return not objective.endswith("-")


def _oriented(values: Sequence[float], objective: str) -> List[float]:
    direction = 1.0 if _better_is_higher(objective) else -1.0
    return [direction * value for value in values]


def certify(
    objective: str,
    treatment: Sequence[float],
    baseline: Sequence[float],
    alpha: float = 0.05,
    permutations: int = 2000,
    seed: int = 42,
) -> Dict[str, Union[float, bool]]:
    """Compare treatment and baseline using an oriented permutation test.

    A positive effect means the treatment is better after considering the
    objective's direction. The test is intentionally dependency-free so it
    can be reused by every treatment.
    """

    if not treatment or not baseline:
        raise ValueError("treatment and baseline must both contain values")
    treatment_o = _oriented(treatment, objective)
    baseline_o = _oriented(baseline, objective)
    observed = mean(treatment_o) - mean(baseline_o)
    pooled = treatment_o + baseline_o
    treatment_size = len(treatment_o)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(permutations):
        shuffled = pooled[:]
        rng.shuffle(shuffled)
        difference = mean(shuffled[:treatment_size]) - mean(shuffled[treatment_size:])
        if abs(difference) >= abs(observed):
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (permutations + 1)
    return {
        "effect": observed,
        "treatment_mean": mean(treatment_o),
        "baseline_mean": mean(baseline_o),
        "p_value": p_value,
        "significant": p_value < alpha,
        "treatment_wins": observed > 0 and p_value < alpha,
    }
