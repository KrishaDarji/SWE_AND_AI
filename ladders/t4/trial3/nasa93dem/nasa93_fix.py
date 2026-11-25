#!/usr/bin/env python3
"""Suggest a minimum-change, training-only repair for an Effort- budget."""
import argparse
import math
from typing import Any, Dict, Mapping, Sequence

from nasa93_effort import TARGET, evaluate, predict_proposal, _value
from nasa93_summary import read_csv


def _budget(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("budget must be finite and nonnegative")
    if isinstance(value, bool) or not math.isfinite(result) or result < 0:
        raise ValueError("budget must be finite and nonnegative")
    return result


def _validate(proposal: Mapping[str, Any], names: Sequence[str], budget: Any) -> float:
    if not isinstance(proposal, Mapping):
        raise ValueError("proposal must be a mapping")
    missing = [name for name in names if name not in proposal]
    if missing:
        raise ValueError("incomplete proposal; missing descriptors: " + ", ".join(missing))
    return _budget(budget)


def _valid_estimate(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0


def repair(result: Mapping[str, Any], proposal: Mapping[str, Any], budget: Any) -> Dict[str, Any]:
    """Repair a complete proposal using only configurations in training rows."""
    names = list(result.get("predictor_names", []))
    limit = _validate(proposal, names, budget)
    original = {name: proposal[name] for name in names}
    original_effort = predict_proposal(result["model"], original, names)
    base = {"original_descriptors": original, "original_estimated_effort": original_effort,
            "budget": limit, "predicted_feasible": _valid_estimate(original_effort) and original_effort <= limit,
            "invalid_candidate_count": 0}
    if not _valid_estimate(original_effort):
        return {**base, "status": "invalid-original-estimate", "candidate_descriptors": None,
                "changed_fields": [], "change_count": None, "candidate_estimated_effort": None,
                "candidate_record_identifier": None}
    if original_effort <= limit:
        return {**base, "status": "no-repair-needed", "candidate_descriptors": original,
                "changed_fields": [], "change_count": 0, "candidate_estimated_effort": original_effort,
                "candidate_record_identifier": None}
    candidates = []
    seen = set()
    for position, row in enumerate(result.get("train_rows", [])):
        candidate = {name: row.get(name) for name in names}
        key = tuple(_value(candidate[name]) for name in names)
        if key in seen:
            continue
        seen.add(key)
        estimate = predict_proposal(result["model"], candidate, names)
        if not _valid_estimate(estimate):
            base["invalid_candidate_count"] += 1
        elif estimate <= limit:
            changed = [name for name in names if _value(candidate[name]) != _value(original[name])]
            candidates.append((len(changed), estimate, position, row, candidate, changed))
    if not candidates:
        return {**base, "status": "no-feasible-repair", "candidate_descriptors": None,
                "changed_fields": [], "change_count": None, "candidate_estimated_effort": None,
                "candidate_record_identifier": None, "predicted_feasible": False}
    count, estimate, position, row, candidate, changed = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {**base, "status": "repair-suggested", "candidate_descriptors": candidate,
            "changed_fields": [{"descriptor": name, "old": original[name], "new": candidate[name]} for name in changed],
            "change_count": count, "candidate_estimated_effort": estimate,
            "candidate_record_identifier": row.get("idX", position + 1), "predicted_feasible": True}


def repair_csv(path: str, budget: Any) -> Dict[str, Any]:
    data = read_csv(path)
    result = evaluate(path)
    selected = result["test_rows"][0]
    proposal = {name: selected.get(name) for name in result["predictor_names"]}
    return {**repair(result, proposal, budget), "proposal_record_identifier": selected.get("idX"),
            "model": result}


def format_report(report: Mapping[str, Any]) -> str:
    lines = ["NASA93 proposal repair", f"Status: {report['status']}",
             f"Proposal record identifier: {report.get('proposal_record_identifier', 'supplied')}",
             f"Budget (Effort-): {report['budget']:.6g}",
             f"Original estimated effort: {report['original_estimated_effort']:.6g}",
             f"Candidate estimated effort: {report['candidate_estimated_effort']!s}",
             f"Predicted feasible: {report['predicted_feasible']}",
             f"Invalid candidate predictions excluded: {report['invalid_candidate_count']}",
             f"Change count (proxy only): {report['change_count']!s}",
             f"Changed fields: {report['changed_fields']!r}",
             f"Original descriptors: {report['original_descriptors']!r}",
             f"Candidate descriptors: {report['candidate_descriptors']!r}",
             "Candidates are hypothetical: descriptors may not be actionable, and observed combinations may not preserve project scope.",
             "Change count is a proxy, not monetary cost or operational risk; model differences are not causal improvements or guaranteed savings."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--budget", required=True)
    args = parser.parse_args()
    print(format_report(repair_csv(args.csv_path, args.budget)), end="")


if __name__ == "__main__":
    main()
