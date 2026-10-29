#!/usr/bin/env python3
"""Training-only, auditable justification of one held-out NASA93 estimate."""
import argparse
import math
from typing import Any, Dict

from nasa93_effort import evaluate, predict_proposal, TARGET
from nasa93_blame import rank_associations
from nasa93_summary import read_csv


def justify(path: str) -> Dict[str, Any]:
    data = read_csv(path)
    result = evaluate(path)
    selected = result["test_rows"][0]
    names = result["predictor_names"]
    proposal = {name: selected.get(name) for name in names}
    estimate = predict_proposal(result["model"], proposal, names)
    associations = rank_associations(result["train_rows"], data["columns"])
    selected_position = next(i for i, row in enumerate(data["rows"])
                             if row.get("idX") == selected.get("idX"))
    invalid_before = sum(1 for row in data["rows"][:selected_position]
                         if not isinstance(row.get(TARGET), (int, float)) or isinstance(row.get(TARGET), bool)
                         or not math.isfinite(float(row[TARGET])))
    return {"identifier": selected.get("idX"), "proposal": proposal, "descriptor_count": len(names),
            "estimate": estimate, "training_mean": result["training_mean"],
            "training_median": float(__import__("numpy").median([float(r[TARGET]) for r in result["train_rows"]])),
            "associations": associations, "observed_effort_audit": selected[TARGET],
            "invalid_target_rows_before_selected": invalid_before, "model": result}


def format_justification(report: Dict[str, Any]) -> str:
    assoc = report["associations"]["rankings"][:3]
    lines = ["MODEL ESTIMATE (pre-audit)", f"Selected held-out identifier: {report['identifier']}",
             f"Proposal inputs: {report['descriptor_count']} descriptors only; metadata and outcomes excluded.",
             f"Training-only Ridge estimate: {report['estimate']:.6g}",
             f"Training-history reference (mean): {report['training_mean']:.6g}",
             f"Training-history reference (median): {float(report['training_median']):.6g}",
             "HISTORICAL ASSOCIATIONS (training rows only; marginal, not additive or causal)"]
    for item in assoc:
        lines.append(f"{item['descriptor']}: eta_squared={item['score']:.6g}; {item['direction']}")
    lines += ["LIMITS", "Missing/unseen descriptor categories are imputed or ignored by the fitted pipeline.",
              "Correlated descriptors may share explanatory information; associations are not local contributions.",
              "This historical estimate does not establish project scope, units, costs, intervals, or deployment guarantees.",
              "AUDIT (observed held-out value revealed afterward)",
              f"Observed Effort-: {report['observed_effort_audit']!r}",
              f"Invalid-target rows before selected CSV record: {report['invalid_target_rows_before_selected']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    print(format_justification(justify(parser.parse_args().csv_path)), end="")


if __name__ == "__main__":
    main()
