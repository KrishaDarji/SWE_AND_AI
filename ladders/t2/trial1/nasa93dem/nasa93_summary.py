#!/usr/bin/env python3
"""Reusable historical-project summary for nasa93dem.csv (stdlib only)."""

import argparse
import csv
import json
import math
import statistics
import random
from dataclasses import asdict, dataclass
from pathlib import Path


OUTCOME_FIELDS = ("Kloc+", "Effort-", "Defects-", "MonthsX")


@dataclass(frozen=True)
class OutcomeSummary:
    field: str
    count: int
    minimum: float
    median: float
    mean: float
    maximum: float


def _percentile(values, fraction):
    """Linear-interpolated percentile using only the standard library."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _finite_number(value):
    try:
        number = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_projects(path):
    """Load rows and validate the columns needed by later skills."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        missing = [name for name in OUTCOME_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError("missing outcome columns: " + ", ".join(missing))
        projects = []
        for line_number, row in enumerate(reader, start=2):
            parsed = {name: _finite_number(row.get(name)) for name in OUTCOME_FIELDS}
            if any(value is None for value in parsed.values()):
                raise ValueError("invalid outcome value on CSV line %d" % line_number)
            parsed["id"] = row.get("idX", "").strip() or str(line_number - 1)
            projects.append(parsed)
    if not projects:
        raise ValueError("CSV contains no projects")
    return projects


def _stats(values, field):
    return OutcomeSummary(field, len(values), min(values), statistics.median(values),
                           statistics.fmean(values), max(values))


def summarize_projects(path):
    """Return a JSON-serializable summary and row-level delivery indicators."""
    projects = load_projects(path)
    summaries = [_stats([project[field] for project in projects], field)
                 for field in OUTCOME_FIELDS]
    enriched = []
    for project in projects:
        item = dict(project)
        item["effort_per_kloc"] = project["Effort-"] / project["Kloc+"] if project["Kloc+"] else None
        item["defects_per_kloc"] = project["Defects-"] / project["Kloc+"] if project["Kloc+"] else None
        item["months_per_kloc"] = project["MonthsX"] / project["Kloc+"] if project["Kloc+"] else None
        enriched.append(item)
    return {"dataset": str(path), "project_count": len(projects),
            "outcomes": [asdict(summary) for summary in summaries],
            "projects": enriched}


def estimate_effort(path, kloc, budget=None):
    """Estimate effort for a proposed project from historical effort/KLOC.

    The median historical effort intensity is the point estimate.  The 25th
    and 75th percentiles form a planning range; approval requires a supplied
    budget to cover the upper planning estimate.
    """
    if not math.isfinite(float(kloc)) or float(kloc) <= 0:
        raise ValueError("kloc must be a positive finite number")
    if budget is not None and (not math.isfinite(float(budget)) or float(budget) < 0):
        raise ValueError("budget must be a non-negative finite number")
    projects = load_projects(path)
    intensities = [p["Effort-"] / p["Kloc+"] for p in projects if p["Kloc+"] > 0]
    median_rate = statistics.median(intensities)
    low_rate = _percentile(intensities, 0.25)
    high_rate = _percentile(intensities, 0.75)
    result = {
        "dataset": str(path), "kloc": float(kloc),
        "historical_projects": len(projects),
        "effort": float(kloc) * median_rate,
        "planning_low": float(kloc) * low_rate,
        "planning_high": float(kloc) * high_rate,
        "effort_per_kloc_median": median_rate,
        "budget": None if budget is None else float(budget),
        "approval": "needs_budget_review" if budget is None else
                    ("approve" if float(budget) >= float(kloc) * high_rate else "reject"),
    }
    return result


def compare_costs(path, group_field, group_a, group_b, samples=4000, seed=0):
    """Assess whether group A is genuinely less costly than group B.

    Cost is effort per KLOC.  The conclusion requires both a lower observed
    median and a one-sided permutation p-value below .05; the bootstrap
    interval reports uncertainty in the median difference (A minus B).
    """
    if samples < 100 or not isinstance(samples, int):
        raise ValueError("samples must be an integer of at least 100")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or group_field not in reader.fieldnames:
            raise ValueError("unknown grouping column: " + group_field)
        groups = {group_a: [], group_b: []}
        for row in reader:
            label = row.get(group_field, "").strip()
            if label in groups:
                kloc = _finite_number(row.get("Kloc+"))
                effort = _finite_number(row.get("Effort-"))
                if kloc is None or effort is None or kloc <= 0:
                    raise ValueError("invalid cost data")
                groups[label].append(effort / kloc)
    a, b = groups[group_a], groups[group_b]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each process group needs at least two projects")
    observed = statistics.median(a) - statistics.median(b)
    combined = a + b
    rng = random.Random(seed)
    more_extreme = 0
    for _ in range(samples):
        shuffled = combined[:]
        rng.shuffle(shuffled)
        if statistics.median(shuffled[:len(a)]) - statistics.median(shuffled[len(a):]) <= observed:
            more_extreme += 1
    p_value = (more_extreme + 1) / (samples + 1)
    boot = []
    for _ in range(samples):
        boot.append(statistics.median(rng.choice(a) for _ in a) -
                    statistics.median(rng.choice(b) for _ in b))
    boot.sort()
    low = _percentile(boot, .025)
    high = _percentile(boot, .975)
    return {"group_field": group_field, "group_a": group_a, "group_b": group_b,
            "count_a": len(a), "count_b": len(b), "median_cost_a": statistics.median(a),
            "median_cost_b": statistics.median(b), "median_difference_a_minus_b": observed,
            "bootstrap_95_low": low, "bootstrap_95_high": high, "permutation_p_value": p_value,
            "genuinely_less_costly": observed < 0 and p_value < .05}


def assess_proposal(path, proposal):
    """Assess one arriving proposal using historical planning ranges.

    A proposal must provide ``id`` and positive ``kloc``.  Optional ``budget``
    and ``max_months`` constraints turn the historical estimates into a
    portfolio decision: approve when the upper effort and expected schedule
    fit the supplied constraints, otherwise flag it for review or reject it.
    """
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a mapping")
    proposal_id = str(proposal.get("id", "")).strip()
    if not proposal_id:
        raise ValueError("proposal id is required")
    if "kloc" not in proposal:
        raise ValueError("proposal kloc is required")
    try:
        kloc = float(proposal["kloc"])
    except (TypeError, ValueError):
        raise ValueError("proposal kloc must be numeric")
    budget = proposal.get("budget")
    max_months = proposal.get("max_months")
    for name, value, minimum in (("budget", budget, 0), ("max_months", max_months, 0)):
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError("proposal %s must be numeric" % name)
            if not math.isfinite(numeric) or numeric < minimum:
                raise ValueError("proposal %s must be non-negative and finite" % name)
    estimate = estimate_effort(path, kloc, budget)
    projects = load_projects(path)
    months_rates = [p["MonthsX"] / p["Kloc+"] for p in projects if p["Kloc+"] > 0]
    defect_rates = [p["Defects-"] / p["Kloc+"] for p in projects if p["Kloc+"] > 0]
    expected_months = kloc * statistics.median(months_rates)
    expected_defects = kloc * statistics.median(defect_rates)
    constraints = []
    if budget is not None:
        constraints.append(estimate["approval"] == "approve")
    if max_months is not None:
        constraints.append(expected_months <= float(max_months))
    decision = "approve" if constraints and all(constraints) else (
        "reject" if any(not item for item in constraints) else "needs_review")
    return {"proposal_id": proposal_id, "decision": decision,
            "estimate": estimate, "expected_months": expected_months,
            "expected_defects": expected_defects,
            "constraints_checked": {"budget": budget is not None,
                                     "max_months": max_months is not None}}


def assess_proposals(path, proposals):
    """Assess proposals in arrival order, retaining each proposal's identity."""
    if isinstance(proposals, dict) or isinstance(proposals, (str, bytes)):
        raise ValueError("proposals must be an iterable of mappings")
    return [assess_proposal(path, proposal) for proposal in proposals]


def analyze_effort_factors(path, minimum_level_count=3):
    """Rank project factors associated with excessive effort intensity.

    Excessive effort means effort/KLOC above the dataset median.  For every
    non-outcome column, this reports levels with enough observations and their
    excess-project rate, plus the lift over the overall excess rate.  This is
    descriptive association, not a causal model.
    """
    if not isinstance(minimum_level_count, int) or minimum_level_count < 1:
        raise ValueError("minimum_level_count must be a positive integer")
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        rows = list(reader)
    rates = []
    for row in rows:
        kloc, effort = _finite_number(row.get("Kloc+")), _finite_number(row.get("Effort-"))
        if kloc is None or effort is None or kloc <= 0:
            raise ValueError("invalid effort data")
        rates.append(effort / kloc)
    cutoff = statistics.median(rates)
    overall_excess = sum(rate > cutoff for rate in rates) / len(rates)
    excluded = set(OUTCOME_FIELDS) | {"idX"}
    factors = []
    for field in reader.fieldnames:
        if field in excluded:
            continue
        levels = {}
        for row, rate in zip(rows, rates):
            value = (row.get(field) or "").strip()
            if value:
                levels.setdefault(value, []).append(rate)
        level_results = []
        for value, values in levels.items():
            if len(values) < minimum_level_count:
                continue
            excess_rate = sum(item > cutoff for item in values) / len(values)
            level_results.append({"level": value, "count": len(values),
                                  "median_effort_per_kloc": statistics.median(values),
                                  "excess_rate": excess_rate,
                                  "lift": excess_rate - overall_excess})
        if level_results:
            level_results.sort(key=lambda item: (-abs(item["lift"]), item["level"]))
            factors.append({"factor": field, "levels": level_results,
                            "strongest_level": level_results[0]["level"],
                            "strongest_lift": level_results[0]["lift"]})
    factors.sort(key=lambda item: (-abs(item["strongest_lift"]), item["factor"]))
    return {"definition": "effort_per_kloc above dataset median", "cutoff": cutoff,
            "overall_excess_rate": overall_excess, "project_count": len(rows),
            "factors": factors}


def justify_estimate(path, kloc, budget=None):
    """Build an evidence-backed explanation suitable for a skeptical sponsor.

    The calculation reuses :func:`estimate_effort` and adds transparent
    historical evidence, uncertainty, decision logic, and limitations.  It
    does not imply that the historical association is causal.
    """
    estimate = estimate_effort(path, kloc, budget)
    projects = load_projects(path)
    rates = [p["Effort-"] / p["Kloc+"] for p in projects if p["Kloc+"] > 0]
    median_rate = statistics.median(rates)
    low_rate, high_rate = _percentile(rates, .25), _percentile(rates, .75)
    return {
        "recommendation": estimate["approval"],
        "estimate": estimate,
        "evidence": {
            "historical_projects": len(projects),
            "historical_effort_per_kloc_median": median_rate,
            "historical_effort_per_kloc_planning_low": low_rate,
            "historical_effort_per_kloc_planning_high": high_rate,
            "planning_range_covers_middle": "25th_to_75th_percentile",
        },
        "sponsor_message": (
            "The point estimate is based on the median effort per KLOC from "
            "%d historical projects. The planning range uses the middle 50%% "
            "of observed effort intensity; approval is recommended only when "
            "the stated budget covers its upper bound."
        ) % len(projects),
        "risks_and_limitations": [
            "Historical projects may differ from the proposed project's scope, team, or technology.",
            "The range describes observed effort intensity and is not a confidence interval.",
            "No schedule, staffing, requirements volatility, or contingency adjustment is modeled.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--estimate-kloc", type=float)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--compare-field")
    parser.add_argument("--process-a")
    parser.add_argument("--process-b")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--proposal-id")
    parser.add_argument("--proposal-kloc", type=float)
    parser.add_argument("--proposal-budget", type=float)
    parser.add_argument("--max-months", type=float)
    parser.add_argument("--analyze-factors", action="store_true")
    parser.add_argument("--minimum-level-count", type=int, default=3)
    parser.add_argument("--justify-kloc", type=float)
    parser.add_argument("--justify-budget", type=float)
    args = parser.parse_args(argv)
    result = (justify_estimate(args.csv_path, args.justify_kloc, args.justify_budget)
              if args.justify_kloc is not None else
              analyze_effort_factors(args.csv_path, args.minimum_level_count)
              if args.analyze_factors else
              assess_proposal(args.csv_path, {"id": args.proposal_id,
                                              "kloc": args.proposal_kloc,
                                              "budget": args.proposal_budget,
                                              "max_months": args.max_months})
              if args.proposal_id else
              compare_costs(args.csv_path, args.compare_field, args.process_a, args.process_b, args.samples)
              if args.compare_field else
              estimate_effort(args.csv_path, args.estimate_kloc, args.budget)
              if args.estimate_kloc is not None else summarize_projects(args.csv_path))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
