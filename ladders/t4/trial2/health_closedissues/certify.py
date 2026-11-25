"""Certify a fixed squared-error versus absolute-error comparison."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def _finite(rows: list[dict[str, str]], criterion: str) -> list[float]:
    values = []
    for row in rows:
        if row.get("criterion", "") != criterion:
            continue
        try:
            value = float(row.get("MRE-", ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _welch(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    n_ref, n_cand = len(reference), len(candidate)
    mean_ref, mean_cand = float(np.mean(reference)), float(np.mean(candidate))
    diff = mean_cand - mean_ref
    var_ref = float(np.var(reference, ddof=1)) if n_ref > 1 else float("nan")
    var_cand = float(np.var(candidate, ddof=1)) if n_cand > 1 else float("nan")
    a, b = var_ref / n_ref, var_cand / n_cand
    se2 = a + b
    if se2 == 0:
        df, se, t_stat, p_value = float("inf"), 0.0, (0.0 if diff == 0 else math.copysign(float("inf"), diff)), (1.0 if diff == 0 else 0.0)
        ci = (diff, diff)
    else:
        df = se2 * se2 / (a * a / (n_ref - 1) + b * b / (n_cand - 1))
        se = math.sqrt(se2)
        t_stat = diff / se
        p_value = float(2 * stats.t.sf(abs(t_stat), df))
        margin = float(stats.t.ppf(0.975, df) * se)
        ci = (diff - margin, diff + margin)
    return {"reference_count": n_ref, "candidate_count": n_cand, "reference_mean": mean_ref, "candidate_mean": mean_cand,
            "difference_candidate_minus_reference": diff, "reference_variance": var_ref, "candidate_variance": var_cand,
            "standard_error": se, "welch_satterthwaite_df": df, "confidence_level": 0.95,
            "confidence_interval": list(ci), "test": "two-sided Welch unequal-variance t-test",
            "t_statistic": t_stat, "p_value": p_value}


def certify(path: str | Path, practical_threshold: float | None = None) -> dict[str, Any]:
    """Compare only absolute_error against squared_error on finite MRE- rows."""
    if practical_threshold is not None and (not math.isfinite(practical_threshold) or practical_threshold < 0):
        raise ValueError("practical_threshold must be a finite nonnegative number")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reference, candidate = _finite(rows, "squared_error"), _finite(rows, "absolute_error")
    if len(reference) < 2 or len(candidate) < 2:
        raise ValueError("insufficient finite MRE- observations in one or both fixed criterion groups")
    primary = _welch(reference, candidate)
    if practical_threshold is None:
        practical = {"threshold": None, "importance": "undetermined: no client threshold was supplied"}
    else:
        d = primary["difference_candidate_minus_reference"]
        lo, hi = primary["confidence_interval"]
        practical = {"threshold": practical_threshold,
                     "observed_difference_exceeds_threshold": d <= -practical_threshold,
                     "entire_confidence_interval_favorably_beyond_threshold": hi <= -practical_threshold,
                     "direction_note": "negative differences favor absolute_error because lower MRE- is better"}
    secondary = {}
    for outcome in ("ACC+", "PRED40+"):
        groups = {}
        for criterion in ("squared_error", "absolute_error"):
            vals = []
            for row in rows:
                if row.get("criterion") == criterion:
                    try:
                        v = float(row.get(outcome, ""))
                        if math.isfinite(v): vals.append(v)
                    except (TypeError, ValueError):
                        pass
            groups[criterion] = {"count": len(vals), "mean": (float(np.mean(vals)) if vals else None)}
        secondary[outcome] = groups
    return {"group_definitions": {"reference": "criterion=squared_error", "candidate": "criterion=absolute_error"},
            "primary_outcome": "MRE-", "primary": primary, "practical_effect": practical,
            "secondary_descriptive_outcomes": secondary,
            "interpretation": "Rows are different configurations, not repeated measurements of one configuration. Other settings may confound criterion-family differences; this analysis does not estimate repeated-run noise, establish equivalence, claim causality, or guarantee gains."}


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    print(json.dumps(certify(args.dataset, args.threshold), indent=2, sort_keys=True))
