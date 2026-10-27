"""Reusable historical project summarization package."""

from .model import Project, DatasetSummary
from .loader import load_projects
from .summarize import summarize_projects

__all__ = ["Project", "DatasetSummary", "load_projects", "summarize_projects"]
