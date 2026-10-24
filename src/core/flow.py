"""Step 4: evaluate data incrementally and stop when evidence stabilizes."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List, Sequence

from .datasets import DatasetSpec
from .remember import as_number


def stream_objective(
    spec: DatasetSpec,
    rows: Sequence[Dict[str, str]],
    minimum: int = 20,
    tolerance: float = 0.01,
    patience: int = 5,
) -> Dict[str, object]:
    """Consume rows in order and stop after a stable running estimate.

    Stability is measured as relative change in the running mean. The result
    includes the complete checkpoint history so the stopping decision can be
    audited and plotted.
    """

    if minimum < 1 or patience < 1:
        raise ValueError("minimum and patience must be positive")
    history: Dict[str, List[Dict[str, float]]] = {objective: [] for objective in spec.objectives}
    values: Dict[str, List[float]] = {objective: [] for objective in spec.objectives}
    stable = 0
    stopped_at = len(rows)
    for index, row in enumerate(rows, start=1):
        for objective in spec.objectives:
            if row.get(objective) not in (None, "", "?"):
                values[objective].append(as_number(row[objective]))
                current = mean(values[objective])
                previous = history[objective][-1]["mean"] if history[objective] else current
                change = abs(current - previous) / max(abs(previous), 1e-12)
                history[objective].append({"seen": float(index), "mean": current, "change": change})
        if index >= minimum:
            changes = [history[objective][-1]["change"] for objective in spec.objectives if history[objective]]
            if changes and max(changes) <= tolerance:
                stable += 1
            else:
                stable = 0
            if stable >= patience:
                stopped_at = index
                break
    return {
        "dataset": spec.name,
        "seen": stopped_at,
        "stopped_early": stopped_at < len(rows),
        "history": history,
        "final_means": {objective: mean(values[objective]) for objective in spec.objectives if values[objective]},
    }
