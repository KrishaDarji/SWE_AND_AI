#!/usr/bin/env python3
"""Rank categorical project descriptors by marginal association with Effort-."""
import argparse
import csv
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    from .nasa93_effort import predictor_names
    from .nasa93_summary import read_csv
except ImportError:
    from nasa93_effort import predictor_names
    from nasa93_summary import read_csv

TARGET = "Effort-"


def _finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _label(value: Any) -> str:
    if value is None:
        return "__MISSING__"
    try:
        if not math.isfinite(float(value)):
            return "__MISSING__"
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else "__MISSING__"


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(np.mean(array)),
            "median": float(np.median(array)), "min": float(np.min(array)),
            "max": float(np.max(array))}


def _eta_squared(groups: Sequence[Sequence[float]], overall: float) -> float:
    total = sum((value - overall) ** 2 for group in groups for value in group)
    between = sum(len(group) * (float(np.mean(group)) - overall) ** 2 for group in groups)
    if total <= 0 or not math.isfinite(total):
        return 0.0
    return float(max(0.0, min(1.0, between / total)))


def rank_associations(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> Dict[str, Any]:
    """Return marginal eta-squared rankings, with Effort- summaries per category.

    The primary score is eta-squared: the fraction of observed Effort- variation
    attributable to differences between descriptor-category means. It is not an
    adjusted or causal effect estimate; each descriptor is evaluated separately.
    """
    names = predictor_names(columns)
    usable = [row for row in rows if _finite(row.get(TARGET))]
    if not names or not usable:
        raise ValueError("input needs at least one descriptor and one finite Effort- row")
    overall = float(np.mean([float(row[TARGET]) for row in usable]))
    rankings: List[Dict[str, Any]] = []
    for name in names:
        grouped: Dict[str, List[float]] = {}
        for row in usable:
            grouped.setdefault(_label(row.get(name)), []).append(float(row[TARGET]))
        ordered = sorted(grouped.items(), key=lambda item: item[0])
        groups = [values for _, values in ordered]
        score = _eta_squared(groups, overall)
        summaries = {key: _summary(values) for key, values in ordered}
        means = {key: summary["mean"] for key, summary in summaries.items()}
        keys = sorted(means)
        highest = max(keys, key=lambda key: (means[key], key))
        lowest = min(keys, key=lambda key: (means[key], key))
        if math.isclose(means[highest], means[lowest], rel_tol=0.0, abs_tol=0.0):
            direction = "no directional contrast (equal category means)"
        else:
            direction = "nominal contrast; highest and lowest category means reported"
        rankings.append({"descriptor": name, "score": score, "direction": direction,
                         "highest_mean": {"category": highest, "effort": means[highest]},
                         "lowest_mean": {"category": lowest, "effort": means[lowest]},
                         "group_count": len(ordered), "groups": summaries})
    rankings.sort(key=lambda item: (-item["score"], item["descriptor"]))
    return {"target": TARGET, "row_count": len(usable), "overall_effort": overall,
            "predictors": names, "score_name": "eta_squared_marginal",
            "rankings": rankings}


def format_report(result: Dict[str, Any]) -> str:
    lines = ["NASA93 Effort- association ranking", f"Usable rows: {result['row_count']}",
             f"Overall Effort- mean: {result['overall_effort']:.6g}",
             "Primary score: marginal eta-squared (between-category / total Effort- variation).",
             "Scores are marginal associations; correlated descriptors can share explanatory information.",
             "Higher effort alone does not establish inefficiency; scope differs, and Kloc+ is excluded."]
    for rank, item in enumerate(result["rankings"], 1):
        high, low = item["highest_mean"], item["lowest_mean"]
        lines.append(f"{rank}. {item['descriptor']}: score={item['score']:.6g}; groups={item['group_count']}; direction={item['direction']}; highest={high['category']} ({high['effort']:.6g}); lowest={low['category']} ({low['effort']:.6g})")
        for key, summary in item["groups"].items():
            lines.append(f"   {key}: n={summary['count']}, mean={summary['mean']:.6g}, median={summary['median']:.6g}, min={summary['min']:.6g}, max={summary['max']:.6g}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(format_report(rank_associations(**(lambda data: {"rows": data["rows"], "columns": data["columns"]})(read_csv(args.csv_path)))), end="")


if __name__ == "__main__":
    main()
