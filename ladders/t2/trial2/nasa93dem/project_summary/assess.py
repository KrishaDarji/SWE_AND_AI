"""Assess incoming project proposals against portfolio constraints."""

from dataclasses import dataclass
from typing import Iterable, List

from .estimate import EffortEstimate, estimate_effort
from .model import Project


@dataclass(frozen=True)
class ProposalAssessment:
    """A decision-ready assessment for one proposal arriving in sequence."""

    proposal_id: str
    proposed_kloc: float
    estimated_effort: float
    screening_upper_effort: float
    remaining_budget: float
    remaining_capacity: float
    decision: str
    reasons: List[str]

    def to_dict(self):
        return self.__dict__.copy()


def assess_proposal(projects: Iterable[Project], proposal_id: str,
                    proposed_kloc: float, remaining_budget: float,
                    remaining_capacity: float, neighbors: int = 7) -> ProposalAssessment:
    """Assess a proposal using historical effort and conservative screening bounds."""
    if not proposal_id.strip():
        raise ValueError("proposal ID must not be empty")
    if remaining_budget < 0 or remaining_capacity < 0:
        raise ValueError("remaining budget and capacity must be non-negative")
    estimate = estimate_effort(projects, proposed_kloc, neighbors)
    reasons = []
    if estimate.estimated_effort > remaining_budget:
        reasons.append("estimated effort exceeds remaining budget")
    if estimate.upper_effort > remaining_capacity:
        reasons.append("screening upper effort exceeds remaining capacity")
    decision = "hold" if reasons else "approve"
    return ProposalAssessment(proposal_id, proposed_kloc,
                              estimate.estimated_effort, estimate.upper_effort,
                              remaining_budget, remaining_capacity,
                              decision, reasons)
