"""Registry of thin domain adapters. Adding a domain = one adapter + one line here."""

from .health_closedissues import SPEC as HEALTH
from .nasa93dem import SPEC as NASA93DEM
from .x264 import SPEC as X264

DATASETS = {spec.name: spec for spec in (X264, NASA93DEM, HEALTH)}
