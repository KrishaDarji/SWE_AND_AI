"""Turn the historical estimate into a concise, auditable sponsor justification."""

from dataclasses import dataclass
from typing import Iterable, List

from .assess import assess_proposal
from .blame import blame_factors
from .estimate import estimate_effort
from .model import Project


@dataclass(frozen=True)
class EstimateJustification:
    proposal_id: str
    decision: str
    estimated_effort: float
    lower_effort: float
    upper_effort: float
    historical_projects: int
    evidence: List[str]
    caveats: List[str]
    sponsor_message: str

    def to_dict(self):
        return self.__dict__.copy()


def justify_estimate(projects: Iterable[Project], proposal_id: str,
                     proposed_kloc: float, remaining_budget: float,
                     remaining_capacity: float, neighbors: int = 7,
                     top_factors: int = 3) -> EstimateJustification:
    """Explain an estimate and its decision in language suitable for a sponsor."""
    projects = list(projects)
    if top_factors <= 0:
        raise ValueError("top factors must be positive")
    assessment = assess_proposal(projects, proposal_id, proposed_kloc,
                                 remaining_budget, remaining_capacity, neighbors)
    estimate = estimate_effort(projects, proposed_kloc, neighbors)
    associations = blame_factors(projects, top=top_factors).factors
    evidence = [
        "The central estimate is %.1f effort units from the median effort/KLOC of %d nearest historical projects." %
        (estimate.estimated_effort, estimate.historical_projects),
        "The historical screening range is %.1f to %.1f effort units." %
        (estimate.lower_effort, estimate.upper_effort),
    ]
    if associations:
        evidence.append("Relevant historical risk signals include %s." % ", ".join(
            "%s=%s (lift %.2f)" % (f.factor, f.value, f.lift) for f in associations))
    caveats = ["The range reflects historical analogs, not a guaranteed outcome.",
               "Factor associations are signals, not causal explanations."]
    reasons = "; ".join(assessment.reasons) if assessment.reasons else "no budget or capacity constraint was triggered"
    message = ("Sponsor recommendation: %s proposal %s at %.1f effort units. "
               "The decision reflects %s. "
               "Plan against the %.1f upper screening value and review the stated caveats."
               % (assessment.decision.upper(), proposal_id, assessment.estimated_effort,
                  reasons, assessment.screening_upper_effort))
    return EstimateJustification(proposal_id, assessment.decision,
                                 estimate.estimated_effort, estimate.lower_effort,
                                 estimate.upper_effort, estimate.historical_projects,
                                 evidence, caveats, message)
