"""Step 10: race the shared chooser against random search, judged by `same`."""

from __future__ import annotations

import random
from statistics import median
from typing import Dict, Sequence

from .certify import same
from .choose import choose
from .datasets import DatasetSpec
from .remember import bounds, score


def race(spec: DatasetSpec, rows: Sequence[Dict[str, str]], budget: int = 20, repeats: int = 10) -> Dict[str, object]:
    """Both methods get the same evaluation budget; repeat with different seeds."""

    if budget < 1:
        raise ValueError("budget must be positive")
    budget = min(budget, len(rows))
    full = bounds(spec, rows)
    shared, baseline = [], []
    for seed in range(repeats):
        shared.append(choose(spec, rows, budget=budget, seed=seed)["score"])
        sample = random.Random(seed).sample(list(rows), budget)
        baseline.append(max(score(spec, row, full) for row in sample))
    indistinguishable = same(shared, baseline)
    return {
        "dataset": spec.name,
        "budget": budget,
        "repeats": repeats,
        "shared_median": median(shared),
        "baseline_median": median(baseline),
        "shared_scores": shared,
        "baseline_scores": baseline,
        "same": indistinguishable,
        "shared_wins": not indistinguishable and median(shared) > median(baseline),
    }
