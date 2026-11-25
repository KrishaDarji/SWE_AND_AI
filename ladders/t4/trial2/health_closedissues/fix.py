"""Retrospective, observed-data repair by the smallest improving change."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

CONFIGURATION_FIELDS = ("N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth")
OUTCOME_FIELDS = ("MRE-", "ACC+", "PRED40+")
NUMERIC_FIELDS = set(CONFIGURATION_FIELDS) - {"criterion"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _configuration(row: dict[str, str]) -> dict[str, Any]:
    result = {}
    for field in CONFIGURATION_FIELDS:
        result[field] = _number(row.get(field, "")) if field in NUMERIC_FIELDS else row.get(field, "")
    return result


def _valid_configuration(configuration: dict[str, Any]) -> bool:
    return all((value is not None) if field in NUMERIC_FIELDS else bool(value)
               for field, value in configuration.items())


def _outcomes(row: dict[str, str]) -> dict[str, float | None]:
    return {field: _number(row.get(field, "")) for field in OUTCOME_FIELDS}


def repair(path: str | Path, baseline_record_id: int = 1) -> dict[str, Any]:
    """Choose the lexicographically best observed strict MRE- improvement."""
    if isinstance(baseline_record_id, bool) or not isinstance(baseline_record_id, int) or baseline_record_id < 1:
        raise ValueError("baseline_record_id must be a positive integer")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"status": "empty input", "baseline_record_id": baseline_record_id,
                "candidate_count": 0, "retrospective_observed_data_repair": True}
    if baseline_record_id > len(rows):
        raise ValueError("invalid baseline: record identifier is outside the input")
    baseline_row = rows[baseline_record_id - 1]
    baseline_config = _configuration(baseline_row)
    baseline_outcomes = _outcomes(baseline_row)
    if not _valid_configuration(baseline_config) or baseline_outcomes["MRE-"] is None:
        raise ValueError("invalid baseline: complete configuration and finite MRE- are required")

    candidates = []
    for record_id, row in enumerate(rows, 1):
        config, outcomes = _configuration(row), _outcomes(row)
        if (record_id != baseline_record_id and _valid_configuration(config)
                and all(value is not None for value in outcomes.values())
                and outcomes["MRE-"] < baseline_outcomes["MRE-"]):
            changed = [field for field in CONFIGURATION_FIELDS
                       if (config[field] != baseline_config[field])]
            candidates.append((len(changed), outcomes, record_id, row, config, changed))
    if not candidates:
        return {"status": "no improving candidate", "baseline_record_id": baseline_record_id,
                "baseline": {"record_id": baseline_record_id, "configuration": baseline_config,
                              "outcomes": baseline_outcomes}, "candidate_count": 0,
                "retrospective_observed_data_repair": True,
                "change_count_note": "Change count is a proxy for configuration difference, not measured implementation cost."}
    _, candidate_outcomes, record_id, row, config, changed = min(
        candidates, key=lambda item: (item[0], item[1]["MRE-"], -item[1]["ACC+"],
                                      -item[1]["PRED40+"], item[2]))
    changes = [{"field": field, "old": baseline_config[field], "new": config[field]}
               for field in changed]
    return {"status": "repaired", "retrospective_observed_data_repair": True,
            "baseline": {"record_id": baseline_record_id, "configuration": baseline_config,
                          "outcomes": baseline_outcomes},
            "repaired": {"record_id": record_id, "configuration": config,
                          "outcomes": candidate_outcomes},
            "changed_fields": changes, "change_count": len(changed),
            "change_count_note": "Change count is a proxy for configuration difference, not measured implementation cost.",
            "observed_mre_improvement": baseline_outcomes["MRE-"] - candidate_outcomes["MRE-"],
            "secondary_outcome_tradeoffs": {
                field: (candidate_outcomes[field] - baseline_outcomes[field]
                        if baseline_outcomes[field] is not None else None)
                for field in ("ACC+", "PRED40+")},
            "secondary_outcome_worsening": {
                field: (candidate_outcomes[field] < baseline_outcomes[field]
                        if baseline_outcomes[field] is not None else None)
                for field in ("ACC+", "PRED40+")},
            "secondary_outcome_comparison_note": {
                field: ("compared with baseline"
                        if baseline_outcomes[field] is not None
                        else "unknown: missing or non-finite baseline outcome prevents comparison")
                for field in ("ACC+", "PRED40+")},
            "candidate_count": len(candidates),
            "selection_order": "change count, MRE-, ACC+, PRED40+, earliest CSV record"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--baseline-record-id", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(repair(args.dataset, args.baseline_record_id), indent=2, sort_keys=True))
