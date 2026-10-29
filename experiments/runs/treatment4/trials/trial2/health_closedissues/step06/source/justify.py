"""Justify an observed configuration using explicit, deterministic evidence."""
from __future__ import annotations

import csv, math
from pathlib import Path
from typing import Any

CONFIGURATION_FIELDS = ("N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")

def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

def _median(values: list[float]) -> float | None:
    return float(sorted(values)[len(values) // 2]) if values and len(values) % 2 else (float((sorted(values)[len(values)//2-1] + sorted(values)[len(values)//2]) / 2) if values else None)

def justify(path: str | Path) -> dict[str, Any]:
    """Select the best observed row and compare it with finite dataset medians.

    Demonstration policy: lowest observed MRE-, then higher ACC+, then higher
    PRED40+, then earliest CSV record. Non-finite values cannot win a tie-break.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = []
    for record_id, row in enumerate(rows, 1):
        values = {outcome: _finite(row.get(outcome, "")) for outcome in OUTCOMES}
        if values["MRE-"] is not None:
            usable.append((record_id, row, values))
    if not usable:
        raise ValueError("no finite MRE- observations available for selection")
    def key(item: tuple[int, dict[str, str], dict[str, float | None]]) -> tuple[float, float, float, int]:
        record_id, _, values = item
        return (values["MRE-"], -(values["ACC+"] if values["ACC+"] is not None else -math.inf), -(values["PRED40+"] if values["PRED40+"] is not None else -math.inf), record_id)
    selected_id, selected_row, selected_values = min(usable, key=key)
    medians = {}
    valid_counts = {}
    for outcome in OUTCOMES:
        values = [v for _, _, metrics in usable if (v := metrics[outcome]) is not None]
        medians[outcome] = _median(values); valid_counts[outcome] = len(values)
    comparison = {outcome: (selected_values[outcome] - medians[outcome] if selected_values[outcome] is not None and medians[outcome] is not None else None) for outcome in OUTCOMES}
    return {
        "evidence_type": "retrospective observed-data evidence",
        "demonstration_policy": "lowest observed MRE-, ties by higher ACC+, then higher PRED40+, then earliest CSV record",
        "row_count": len(rows), "eligible_mre_count": len(usable),
        "selected": {"record_id": selected_id, "configuration": {field: selected_row.get(field, "") for field in CONFIGURATION_FIELDS}, "outcomes": selected_values},
        "dataset_medians": medians, "median_valid_counts": valid_counts,
        "selected_minus_median": comparison,
        "missing_nonfinite": {outcome: sum(_finite(row.get(outcome, "")) is None for row in rows) for outcome in OUTCOMES},
        "tradeoff_note": "Best MRE- does not necessarily mean best ACC+ or PRED40+; all three measured outcomes are reported independently.",
        "association_note": "Associations, if added from prior analysis, are descriptive and not additive local contributions or causal effects. Correlated settings can share explanatory information.",
        "limitation": "This is observed-data evidence and does not guarantee future performance or establish statistical superiority; no model estimates are used.",
    }

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); args = parser.parse_args()
    print(json.dumps(justify(args.dataset), indent=2, sort_keys=True))
