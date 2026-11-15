"""Step 7: choose a strong candidate while evaluating as few as possible.

Reuses step 2 (guess): label a few random rows, then repeatedly evaluate the
unlabelled row whose *predicted* outcome looks best.
"""

from __future__ import annotations

import random
from typing import Dict, Sequence

from .datasets import DatasetSpec
from .guess import predict_row
from .remember import bounds, score


def choose(
    spec: DatasetSpec,
    rows: Sequence[Dict[str, str]],
    budget: int = 20,
    start: int = 4,
    seed: int = 1,
) -> Dict[str, object]:
    """Pick the best row after evaluating only `budget` rows."""

    if budget < 1:
        raise ValueError("budget must be positive")
    if not rows:
        raise ValueError("rows must not be empty")
    budget = min(budget, len(rows))
    pool = list(rows)
    random.Random(seed).shuffle(pool)
    labelled, pool = pool[: min(start, budget)], pool[min(start, budget):]
    while len(labelled) < budget:
        limits = bounds(spec, labelled)
        guesses = [score(spec, predict_row(spec, labelled, row, k=3), limits) for row in pool]
        best = max(range(len(pool)), key=lambda i: guesses[i] if guesses[i] == guesses[i] else -1)
        labelled.append(pool.pop(best))
    full = bounds(spec, rows)  # used only to report quality, never to guide the search
    selected = max(labelled, key=lambda row: score(spec, row, full))
    selected_score = score(spec, selected, full)
    return {
        "dataset": spec.name,
        "budget": budget,
        "available_rows": len(rows),
        "evaluated": len(labelled),
        "coverage": len(labelled) / len(rows),
        "score": selected_score,
        "beats_fraction_of_all_rows": sum(score(spec, row, full) <= selected_score for row in rows) / len(rows),
        "selected": selected,
    }
