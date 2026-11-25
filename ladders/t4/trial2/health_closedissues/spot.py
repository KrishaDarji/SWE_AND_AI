"""Robust marginal screen for unusual observed quality outcomes."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

OUTCOMES = ("MRE-", "ACC+", "PRED40+")
DEFAULT_THRESHOLD = 3.5


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _screen(outcome: str, records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    values = [(_finite(row.get(outcome)), i + 1, row) for i, row in enumerate(records)]
    finite = [value for value, _, _ in values if value is not None]
    if not finite:
        return {"outcome": outcome, "status": "no_finite_observations", "reason": "No finite observations were available.",
                "reference_includes_assessed_record": True, "reference_count": 0, "median": None,
                "mad": None, "robust_scale": None, "threshold": threshold, "flags": [], "flag_count": 0}
    median = float(np.median(finite)); mad = float(np.median(np.abs(np.asarray(finite) - median)))
    scale = 1.4826 * mad
    flags = []
    for value, record_id, row in values:
        if value is None:
            continue
        deviation = value - median
        if scale > 0:
            score = deviation / scale
            flagged = abs(score) >= threshold
            reason = "absolute robust score meets threshold" if flagged else "below threshold"
        else:
            score = None if deviation else 0.0
            flagged = bool(deviation)
            reason = ("non-median value with zero robust scale; investigation candidate with unbounded score"
                      if flagged else "equals median and zero robust scale")
        if flagged:
            flags.append({"record_id": record_id, "observed_value": value, "reference_median": median,
                          "score": score, "threshold": threshold, "reason": reason})
    return {"outcome": outcome, "status": "ok", "reason": "Finite observations screened with median and MAD.",
            "reference_includes_assessed_record": True, "reference_count": len(finite), "median": median,
            "mad": mad, "robust_scale": scale, "threshold": threshold, "flags": flags, "flag_count": len(flags)}


def screen_records(records: Iterable[dict[str, Any]], threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Screen all three outcomes; flags are candidates, not confirmed errors."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative number")
    rows = list(records)
    outcomes = [_screen(outcome, rows, threshold) for outcome in OUTCOMES]
    flagged_ids = sorted({flag["record_id"] for result in outcomes for flag in result["flags"]})
    return {"status": "ok", "method": "signed robust z-score: (observed - median) / (1.4826 * MAD)",
            "threshold_rule": "flag when absolute score is at least threshold; zero-scale non-median values are flagged with score null",
            "threshold": threshold, "reference_includes_assessed_record": True, "record_count": len(rows),
            "outcomes": outcomes, "unique_flagged_record_count": len(flagged_ids),
            "per_outcome_flag_counts": {result["outcome"]: result["flag_count"] for result in outcomes},
            "flagged_record_ids": flagged_ids,
            "interpretation": "Flags are investigation candidates, not confirmed measurement errors. This marginal screen does not condition on configuration settings and may flag legitimate differences between configurations. No detection accuracy is claimed without verified anomaly labels."}


def screen(path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return screen_records(csv.DictReader(handle), threshold)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("dataset"); parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    print(json.dumps(screen(args.dataset, args.threshold), indent=2, sort_keys=True, allow_nan=False))
