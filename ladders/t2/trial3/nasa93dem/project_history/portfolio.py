"""Arrival-order assessment of software proposals for portfolio planning."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from .estimator import estimate_effort
from .model import Project


@dataclass(frozen=True)
class ProposalAssessment:
    """The decision and evidence recorded for one arriving proposal."""

    proposal_id: str
    proposed_kloc: float
    estimated_effort: Optional[float]
    cumulative_effort: Optional[float]
    status: str


def assess_proposals(projects: Iterable[Project], proposals: Iterable[Mapping[str, object]],
                     *, max_effort: Optional[float] = None,
                     portfolio_budget: Optional[float] = None) -> list[ProposalAssessment]:
    """Assess proposals in arrival order and reserve effort for approvals.

    Each proposal must provide ``id`` and ``kloc``.  A proposal is approved only
    when its estimate meets the per-proposal limit and the running approved
    total meets the portfolio budget.  Missing estimates are marked unknown.
    """
    if max_effort is not None and max_effort < 0:
        raise ValueError("max_effort must not be negative")
    if portfolio_budget is not None and portfolio_budget < 0:
        raise ValueError("portfolio_budget must not be negative")
    history = list(projects)
    reserved = 0.0
    results = []
    for position, proposal in enumerate(proposals, 1):
        proposal_id = str(proposal.get("id", position))
        try:
            kloc = float(proposal["kloc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("each proposal needs a numeric kloc") from error
        estimate = estimate_effort(history, kloc)
        if estimate is None:
            status = "cannot assess"
            cumulative = None
        else:
            within_item = max_effort is None or estimate <= max_effort
            within_portfolio = (portfolio_budget is None or
                                reserved + estimate <= portfolio_budget)
            if within_item and within_portfolio:
                reserved += estimate
                status = "approved"
            else:
                status = "not approved"
            cumulative = reserved
        results.append(ProposalAssessment(proposal_id, kloc, estimate, cumulative, status))
    return results
