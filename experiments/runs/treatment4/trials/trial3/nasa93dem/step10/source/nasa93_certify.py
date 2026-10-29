#!/usr/bin/env python3
"""Certify an observed Effort- difference between NASA93 tool groups."""
import argparse
import math
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from scipy.stats import t

try:
    from .nasa93_summary import read_csv
except ImportError:
    from nasa93_summary import read_csv


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _threshold(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("threshold must be finite and nonnegative")
    if not math.isfinite(result) or result < 0:
        raise ValueError("threshold must be finite and nonnegative")
    return result


def certify(rows: Iterable[Mapping[str, Any]], threshold: float) -> Dict[str, Any]:
    threshold = _threshold(threshold)
    groups = {"A": [], "B": []}
    for row in rows:
        tool = str(row.get("tool", "")).strip().lower()
        effort = row.get("Effort-")
        if tool in ("n", "h") and _finite(effort):
            groups["A" if tool == "n" else "B"].append(float(effort))
    if not groups["A"] or not groups["B"]:
        raise ValueError("both tool=n (A) and tool=h (B) need finite outcomes")
    a, b = np.asarray(groups["A"]), np.asarray(groups["B"])
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each group needs at least two finite outcomes")
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    diff = mean_a - mean_b
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se2 = var_a / len(a) + var_b / len(b)
    if not math.isfinite(se2) or se2 <= 0:
        raise ValueError("Welch interval is undefined when both groups have zero variance")
    df = se2 ** 2 / ((var_a / len(a)) ** 2 / (len(a) - 1) + (var_b / len(b)) ** 2 / (len(b) - 1))
    se = math.sqrt(se2)
    critical = float(t.ppf(0.975, df))
    interval = (diff - critical * se, diff + critical * se)
    statistic = diff / se
    p_value = float(2 * t.sf(abs(statistic), df))
    return {"group_a": "tool=n", "group_b": "tool=h", "n_a": len(a), "n_b": len(b),
            "mean_a": mean_a, "mean_b": mean_b, "difference_a_minus_b": diff,
            "variance_a": var_a, "variance_b": var_b, "degrees_of_freedom": df,
            "standard_error": se, "confidence_interval_95": interval,
            "welch_t": statistic, "p_value_two_sided": p_value, "threshold": threshold,
            "statistically_different": p_value < 0.05,
            "point_exceeds_threshold": abs(diff) > threshold,
            "interval_above_threshold": interval[0] > threshold,
            "interval_below_negative_threshold": interval[1] < -threshold,
            "practical_conclusion": ("favors B" if interval[0] > threshold else "favors A" if interval[1] < -threshold else "uncertain")}


def certify_csv(path: str, threshold: float) -> Dict[str, Any]:
    return certify(read_csv(path)["rows"], threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    result = certify_csv(args.csv_path, args.threshold)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
