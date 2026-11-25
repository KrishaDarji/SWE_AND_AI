"""Reusable historical project summarization package."""

from .model import Project, DatasetSummary
from .loader import load_projects
from .summarize import summarize_projects
from .estimate import EffortEstimate, estimate_effort
from .certify import CostComparison, certify_less_costly
from .assess import ProposalAssessment, assess_proposal

__all__ = ["Project", "DatasetSummary", "load_projects", "summarize_projects",
           "EffortEstimate", "estimate_effort", "CostComparison", "certify_less_costly",
           "ProposalAssessment", "assess_proposal"]
