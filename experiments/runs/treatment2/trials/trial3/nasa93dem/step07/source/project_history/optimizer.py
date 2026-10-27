"""Choose a value-maximizing proposal subset under an effort budget."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .estimator import estimate_effort
from .model import Project


@dataclass(frozen=True)
class ValuePlan:
    """The selected proposals and the value/cost they deliver."""

    selected_ids: tuple[str, ...]
    total_value: float
    total_effort: float
    rejected_ids: tuple[str, ...]


def maximize_delivered_value(projects: Iterable[Project], proposals: Iterable[Mapping[str, object]],
                             budget: float) -> ValuePlan:
    """Select a feasible subset with maximum value using exact 0/1 knapsack.

    Ties prefer lower effort, then the proposal order supplied by the caller.
    Proposals with invalid/non-positive size, missing value, or unavailable
    historical estimates are rejected rather than silently treated as free.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")
    history = list(projects)
    candidates = []
    rejected = []
    for position, proposal in enumerate(proposals, 1):
        proposal_id = str(proposal.get("id", position))
        try:
            kloc = float(proposal["kloc"])
            value = float(proposal["value"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each proposal needs numeric kloc and value") from error
        effort = estimate_effort(history, kloc) if kloc > 0 else None
        if effort is None or value < 0:
            rejected.append(proposal_id)
        else:
            candidates.append((proposal_id, effort, value, position))
    states = {0: (0.0, 0.0, ())}
    for proposal_id, effort, value, position in candidates:
        additions = {}
        for used, (old_effort, old_value, ids) in states.items():
            new_effort = used + effort
            if new_effort <= budget + 1e-9:
                candidate = (new_effort, old_value + value, ids + (proposal_id,))
                existing = states.get(round(new_effort, 10))
                key = round(new_effort, 10)
                if existing is None or candidate[1] > existing[1] or (candidate[1] == existing[1] and len(candidate[2]) < len(existing[2])):
                    additions[key] = candidate
        states.update(additions)
    best = max(states.values(), key=lambda item: (item[1], -item[0], tuple(-i for i in range(len(item[2])))))
    selected = set(best[2])
    return ValuePlan(best[2], best[1], best[0], tuple(rejected + [item[0] for item in candidates if item[0] not in selected]))
