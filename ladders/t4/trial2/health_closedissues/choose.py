"""Select the best observed feasible configuration under an estimator cap."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

CONFIGURATION_FIELDS = ("N_estimators", "criterion", "Min_sample_leaves",
                        "Min_impurity_decrease", "Max_depth")
OUTCOMES = ("MRE-", "ACC+", "PRED40+")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _cap(value: Any) -> float:
    number = _finite(value)
    if number is None or number < 0:
        raise ValueError("cap must be a finite nonnegative number")
    return number


def choose(path: str | Path, cap: float | int = 100) -> dict[str, Any]:
    """Choose among observed rows with finite settings/outcomes and N_estimators <= cap."""
    limit = _cap(cap)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feasible = []
    for record_id, row in enumerate(rows, 1):
        settings = {field: _finite(row.get(field, "")) for field in CONFIGURATION_FIELDS
                    if field != "criterion"}
        outcomes = {field: _finite(row.get(field, "")) for field in OUTCOMES}
        if (all(value is not None for value in settings.values()) and
                all(value is not None for value in outcomes.values()) and
                settings["N_estimators"] <= limit):
            feasible.append((record_id, row, settings, outcomes))
    if not feasible:
        raise ValueError("no feasible records under the supplied cap")

    def key(item: tuple[int, dict[str, str], dict[str, float | None], dict[str, float | None]]):
        record_id, _, _, outcomes = item
        return (outcomes["MRE-"], -outcomes["ACC+"], -outcomes["PRED40+"], record_id)

    record_id, row, _, outcomes = min(feasible, key=key)
    return {
        "selection_method": "retrospective selection among observed configurations",
        "compute_proxy": "upper limit on N_estimators; other settings also affect compute",
        "cap": limit, "feasible_count": len(feasible), "total_count": len(rows),
        "selected": {"record_id": record_id,
                      "configuration": {field: row.get(field, "") for field in CONFIGURATION_FIELDS},
                      "outcomes": outcomes},
        "observed_outcomes_note": "Selection uses observed outcomes, not predictions for unseen settings.",
        "limitation": "The N_estimators cap is only a compute proxy; it is not a measured runtime, memory, or monetary cost and does not guarantee future quality or feasibility.",
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--cap", type=float, default=100)
    args = parser.parse_args()
    print(json.dumps(choose(args.dataset, args.cap), indent=2, sort_keys=True))
