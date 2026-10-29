#!/usr/bin/env python3
"""Robust marginal screening of unusual NASA93 outcome values."""
import argparse
import json
import math
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import RobustScaler

OUTCOMES = ("Kloc+", "Effort-", "Defects-")
DEFAULT_THRESHOLD = 3.5


def _finite(value: Any) -> bool:
    try:
        return value is not None and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any):
    if not _finite(value):
        return None
    number = float(value)
    return number if number >= 0 else None


def _identifier(row: Mapping[str, Any], position: int) -> Any:
    value = row.get("idX", position + 1)
    return value if value is not None else position + 1


def screen(rows: Iterable[Mapping[str, Any]], columns: Sequence[str], threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Screen each outcome using log1p and median/MAD robust scores.

    Reference statistics include every valid assessed record, including the record
    being reported. Invalid, non-finite, and negative values are excluded but retain
    their original CSV identifiers. A zero MAD yields an undefined score; a value
    differing from the center is nevertheless flagged as a zero-scale candidate.
    """
    if not _finite(threshold) or float(threshold) < 0:
        raise ValueError("threshold must be finite and nonnegative")
    threshold = float(threshold)
    material = list(rows)
    fields = [name for name in OUTCOMES if name in columns]
    records = []
    per_field = {name: 0 for name in fields}
    valid_values = {}
    references = {}
    for name in fields:
        valid = [(i, _number(row.get(name))) for i, row in enumerate(material)]
        valid = [(i, value) for i, value in valid if value is not None]
        transformed = np.log1p(np.asarray([value for _, value in valid], dtype=float))
        if len(transformed):
            center = float(np.median(transformed))
            mad = float(np.median(np.abs(transformed - center)))
            scale = float(1.4826 * mad)
            valid_values[name] = {i: value for i, value in valid}
        else:
            center = mad = scale = None
        references[name] = {"valid_count": len(valid), "log1p_median": center,
                            "mad": mad, "robust_scale": scale,
                            "includes_assessed_record": True}
    for position, row in enumerate(material):
        flags = []
        details = {}
        for name in fields:
            value = _number(row.get(name))
            ref = references[name]
            detail = {"observed": value, "reference": ref, "score": None,
                      "threshold": threshold, "flagged": False, "reason": None}
            if value is None:
                detail["reason"] = "missing, non-finite, or negative value excluded from reference"
            elif ref["robust_scale"] is None:
                detail["reason"] = "no valid reference values; score undefined"
            elif ref["robust_scale"] == 0:
                if not math.isclose(math.log1p(value), ref["log1p_median"], rel_tol=0, abs_tol=0):
                    detail["flagged"] = True; flags.append(name); per_field[name] += 1
                detail["reason"] = "zero robust scale; non-center values are candidates" if detail["flagged"] else "zero robust scale; value equals center"
            else:
                score = abs(math.log1p(value) - ref["log1p_median"]) / ref["robust_scale"]
                detail["score"] = float(score)
                detail["flagged"] = score > threshold
                detail["reason"] = "absolute robust log1p score exceeds threshold" if detail["flagged"] else "within threshold"
                if detail["flagged"]:
                    flags.append(name); per_field[name] += 1
            details[name] = detail
        records.append({"record_identifier": _identifier(row, position), "flagged_fields": flags,
                        "flagged": bool(flags), "outcomes": details})
    return {"method": "absolute robust z-score on log1p outcome, median and 1.4826*MAD",
            "threshold": threshold, "reference_statistics_include_assessed_record": True,
            "outcomes": fields, "records": records,
            "unique_flagged_record_count": sum(item["flagged"] for item in records),
            "per_outcome_flag_counts": per_field,
            "interpretation": "Flags are investigation candidates, not confirmed errors. Large projects may legitimately have large effort or defect counts; this marginal screen is not scope-adjusted, accuracy-validated, or a prediction-uncertainty estimate."}


def screen_csv(path: str, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    from nasa93_summary import read_csv
    data = read_csv(path)
    return screen(data["rows"], data["columns"], threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    print(json.dumps(screen_csv(args.csv_path, args.threshold), sort_keys=True, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
