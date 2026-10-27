"""Sponsor-facing, evidence-based justification for an effort estimate."""

from statistics import median
from typing import Iterable, Optional

from .model import Project


def justify_estimate(projects: Iterable[Project], proposed_kloc: float) -> dict[str, object]:
    """Build a concise justification from historical effort-per-KLOC evidence.

    The estimate is the existing median-productivity estimate.  The spread is
    reported as an interquartile range so a sponsor can see uncertainty; this
    function does not claim that historical association proves causation.
    """
    if proposed_kloc <= 0:
        raise ValueError("proposed KLOC must be greater than zero")
    productivity = sorted(
        item.effort / item.kloc
        for item in projects
        if item.effort is not None and item.kloc is not None and item.kloc > 0
    )
    if not productivity:
        return {"estimate": None, "proposed_kloc": proposed_kloc, "benchmark_count": 0,
                "median_productivity": None, "lower_effort": None, "upper_effort": None,
                "confidence": "not supportable", "caveats": ["No historical records contain both positive KLOC and effort."]}
    midpoint = median(productivity)
    lower = productivity[(len(productivity) - 1) // 4]
    upper = productivity[(3 * (len(productivity) - 1)) // 4]
    return {
        "estimate": proposed_kloc * midpoint,
        "proposed_kloc": proposed_kloc,
        "benchmark_count": len(productivity),
        "median_productivity": midpoint,
        "lower_effort": proposed_kloc * lower,
        "upper_effort": proposed_kloc * upper,
        "confidence": "planning baseline, not a guarantee",
        "caveats": [
            "The range reflects historical productivity spread, not a formal prediction interval.",
            "The comparison is descriptive and does not establish that size alone causes effort.",
            "Actual results may differ because project conditions and missing values are not fully represented.",
        ],
    }


def justification_lines(result: dict[str, object]) -> list[str]:
    """Render a stable explanation suitable for a sponsor discussion."""
    if result["estimate"] is None:
        return ["Estimate justification: unavailable", "Evidence: no usable historical benchmarks"]
    return [
        f"Estimate justification: {result['estimate']:.2f} person-months",
        f"Evidence: {result['benchmark_count']} historical projects with usable size and effort",
        f"Method: proposed KLOC x median effort/KLOC ({result['median_productivity']:.2f})",
        f"Historical productivity range translated to this size: {result['lower_effort']:.2f}-{result['upper_effort']:.2f} person-months",
        f"Confidence: {result['confidence']}",
        "Caveats: historical spread is uncertainty evidence, not a guarantee or causal proof",
    ]
