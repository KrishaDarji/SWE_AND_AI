"""Step 10: compare the shared chooser with a simple baseline."""

from __future__ import annotations

from typing import Dict, Sequence

from .choose import choose
from .datasets import DatasetSpec
from .fix import _score
from .remember import rank_rows


def race(spec: DatasetSpec, rows: Sequence[Dict[str, str]], budget: int = 20) -> Dict[str, object]:
    """Race the shared chooser against a first-rows baseline.

    Both methods receive the same number of candidate evaluations. Scores are
    evaluated against the full dataset objective ranges after selection.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    budget = min(budget, len(rows))
    shared = choose(spec, rows, budget=budget)
    baseline_candidates = list(rows[:budget])
    baseline = rank_rows(spec, baseline_candidates, top_k=1)[0][1]
    shared_score = _score(spec, shared["selected"], rows)
    baseline_score = _score(spec, baseline, rows)
    return {
        "dataset": spec.name,
        "budget": budget,
        "shared_score": shared_score,
        "baseline_score": baseline_score,
        "improvement": shared_score - baseline_score,
        "shared_wins": shared_score > baseline_score,
        "shared_selected": shared["selected"],
        "baseline_selected": baseline,
    }
