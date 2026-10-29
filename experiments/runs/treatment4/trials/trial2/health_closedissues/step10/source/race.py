"""Retrospective comparison of the Step 7 selector with uniform baseline draws."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from choose import CONFIGURATION_FIELDS, OUTCOMES, _cap, _finite

TIE_TOLERANCE = 1e-12
SEEDS = tuple(range(42, 52))


def eligible_records(path: str | Path, cap: float | int = 100) -> list[dict[str, Any]]:
    """Load records using exactly the selector's field-validity and cap rules."""
    limit = _cap(cap)
    records = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record_id, row in enumerate(csv.DictReader(handle), 1):
            numeric = {field: _finite(row.get(field, "")) for field in CONFIGURATION_FIELDS
                       if field != "criterion"}
            outcomes = {field: _finite(row.get(field, "")) for field in OUTCOMES}
            if (all(value is not None for value in numeric.values()) and
                    all(value is not None for value in outcomes.values()) and
                    numeric["N_estimators"] <= limit):
                records.append({"record_id": record_id,
                                "configuration": {field: row.get(field, "") for field in CONFIGURATION_FIELDS},
                                "outcomes": outcomes})
    if not records:
        raise ValueError("no feasible records under the supplied cap")
    return records


def _draw(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    # A new RNG per seed makes draws independent and reproducible.
    return random.Random(seed).choice(records)


def compare(path: str | Path, cap: float | int = 100,
            seeds: tuple[int, ...] = SEEDS,
            tie_tolerance: float = TIE_TOLERANCE) -> dict[str, Any]:
    """Compare the deterministic selector with one uniform draw per seed."""
    from choose import choose
    selected = choose(path, cap)["selected"]
    records = eligible_records(path, cap)
    trials = []
    for seed in seeds:
        baseline = _draw(records, seed)
        difference = baseline["outcomes"]["MRE-"] - selected["outcomes"]["MRE-"]
        trials.append({"seed": seed, "selector_record_id": selected["record_id"],
                       "baseline_record_id": baseline["record_id"],
                       "selector_configuration": selected["configuration"],
                       "baseline_configuration": baseline["configuration"],
                       "selector_outcomes": selected["outcomes"],
                       "baseline_outcomes": baseline["outcomes"],
                       "mre_difference": difference})
    differences = [trial["mre_difference"] for trial in trials]
    # Difference is baseline MRE- minus selector MRE-: positive is a selector win.
    wins = sum(value > tie_tolerance for value in differences)
    ties = sum(abs(value) <= tie_tolerance for value in differences)
    losses = sum(value < -tie_tolerance for value in differences)
    mean = sum(differences) / len(differences)
    ordered = sorted(differences)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered)//2-1] + ordered[len(ordered)//2]) / 2
    return {"cap": float(cap), "eligible_count": len(records), "selector": selected,
            "tie_tolerance": tie_tolerance, "trials": trials,
            "mre_difference_summary": {"mean": mean, "median": median, "min": min(differences), "max": max(differences)},
            "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses},
            "interpretation": "Difference is baseline MRE- minus selector MRE-. Positive beyond tolerance is a selector win, negative beyond tolerance is a selector loss, and within tolerance is a tie. These are retrospective draws from one historical dataset, not independent validation samples or independent LLM trials. The selector inspects all feasible observed outcomes while the baseline samples one record; a win is not evidence of superior budgeted search. ACC+ and PRED40+ are descriptive trade-offs. The estimator cap is a compute proxy, not measured cost. Final project same procedure analysis is pending if unavailable."}
