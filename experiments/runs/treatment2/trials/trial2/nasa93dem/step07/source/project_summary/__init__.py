"""Reusable historical project summarization package."""

from .model import Project, DatasetSummary
from .loader import load_projects
from .summarize import summarize_projects
from .estimate import EffortEstimate, estimate_effort
from .certify import CostComparison, certify_less_costly
from .assess import ProposalAssessment, assess_proposal
from .blame import FactorAssociation, BlameReport, blame_factors
from .justify import EstimateJustification, justify_estimate
from .optimize import SettingRecommendation, OptimizationResult, choose_settings

__all__ = ["Project", "DatasetSummary", "load_projects", "summarize_projects",
           "EffortEstimate", "estimate_effort", "CostComparison", "certify_less_costly",
           "ProposalAssessment", "assess_proposal", "FactorAssociation", "BlameReport",
           "blame_factors", "EstimateJustification", "justify_estimate",
           "SettingRecommendation", "OptimizationResult", "choose_settings"]
