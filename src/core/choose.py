"""Step 7: choose strong candidates under a limited evaluation budget."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .datasets import DatasetSpec
from .remember import rank_rows


def choose(
    spec: DatasetSpec,
    rows: Sequence[Dict[str, str]],
    budget: int = 20,
) -> Dict[str, object]:
    """Select the best observed candidate from a deterministic budget sample."""

    if budget < 1:
        raise ValueError("budget must be positive")
    if not rows:
        raise ValueError("rows must not be empty")
    budget = min(budget, len(rows))
    stride = max(1, len(rows) // budget)
    candidates = list(rows[::stride])[:budget]
    ranked = rank_rows(spec, candidates, top_k=1)
    score, row = ranked[0]
    return {
        "dataset": spec.name,
        "budget": budget,
        "available_rows": len(rows),
        "evaluated": len(candidates),
        "coverage": len(candidates) / len(rows),
        "score": score,
        "selected": row,
    }
