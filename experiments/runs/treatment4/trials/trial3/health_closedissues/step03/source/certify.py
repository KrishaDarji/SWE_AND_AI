"""Certify whether the fixed criterion comparison is distinguishable from noise."""
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

REFERENCE = "squared_error"
CANDIDATE = "absolute_error"
PRIMARY = "MRE-"
SECONDARY = ("ACC+", "PRED40+")


def _finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _values(rows, criterion, field):
    return np.asarray([v for row in rows if row.get("criterion", "").strip() == criterion
                       for v in [_finite(row.get(field))] if v is not None], dtype=float)


def _welch(reference, candidate, confidence=0.95):
    if len(reference) < 2 or len(candidate) < 2:
        raise ValueError("each comparison group needs at least two finite observations")
    n0, n1 = len(reference), len(candidate)
    mean0, mean1 = float(np.mean(reference)), float(np.mean(candidate))
    var0, var1 = float(np.var(reference, ddof=1)), float(np.var(candidate, ddof=1))
    se2 = var0 / n0 + var1 / n1
    difference = mean1 - mean0
    if se2 == 0:
        if difference == 0:
            return {"difference": difference, "standard_error": 0.0, "degrees_of_freedom": None,
                    "t_statistic": 0.0, "p_value_two_sided": 1.0, "ci95": [0.0, 0.0],
                    "zero_variance": True}
        return {"difference": difference, "standard_error": 0.0, "degrees_of_freedom": None,
                "t_statistic": math.copysign(math.inf, difference), "p_value_two_sided": 0.0,
                "ci95": [difference, difference], "zero_variance": True}
    df = se2 ** 2 / ((var0 / n0) ** 2 / (n0 - 1) + (var1 / n1) ** 2 / (n1 - 1))
    se = math.sqrt(se2)
    critical = float(stats.t.ppf(0.975, df))
    t_stat = difference / se
    return {"difference": difference, "standard_error": se, "degrees_of_freedom": float(df),
            "t_statistic": float(t_stat), "p_value_two_sided": float(2 * stats.t.sf(abs(t_stat), df)),
            "ci95": [difference - critical * se, difference + critical * se], "zero_variance": False}


def compare(path, practical_threshold=None):
    if practical_threshold is not None and (not isinstance(practical_threshold, (int, float))
                                             or not math.isfinite(practical_threshold) or practical_threshold < 0):
        raise ValueError("practical_threshold must be finite and nonnegative")
    rows = _read(path)
    result = {"definitions": {"primary": PRIMARY, "reference": REFERENCE, "candidate": CANDIDATE,
                               "difference": "candidate minus reference", "lower_is_better": True,
                               "confidence_level": 0.95, "test": "two-sided Welch unequal-variance"},
              "groups": {}, "secondary": {}, "practical_threshold": practical_threshold}
    ref = _values(rows, REFERENCE, PRIMARY); cand = _values(rows, CANDIDATE, PRIMARY)
    result["groups"] = {"reference_count": int(len(ref)), "candidate_count": int(len(cand)),
                         "reference_mean": float(np.mean(ref)) if len(ref) else None,
                         "candidate_mean": float(np.mean(cand)) if len(cand) else None}
    result["primary"] = _welch(ref, cand)
    if practical_threshold is None:
        result["practical_importance"] = "undetermined: no threshold supplied"
    else:
        d = result["primary"]["difference"]; lo, hi = result["primary"]["ci95"]
        result["practical_importance"] = {"point_estimate_beyond_threshold": d <= -practical_threshold,
                                           "entire_interval_beyond_threshold": hi <= -practical_threshold}
    for field in SECONDARY:
        a, b = _values(rows, REFERENCE, field), _values(rows, CANDIDATE, field)
        result["secondary"][field] = {"reference_count": len(a), "candidate_count": len(b),
                                       "reference_mean": float(np.mean(a)) if len(a) else None,
                                       "candidate_mean": float(np.mean(b)) if len(b) else None,
                                       "candidate_minus_reference": float(np.mean(b)-np.mean(a)) if len(a) and len(b) else None}
    result["interpretation"] = "Negative differences favor the candidate; configurations are different observations, not repeated measurements, and other settings may confound differences. No causal effect, repeated-run noise estimate, or guaranteed gain is claimed."
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset"); parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    print(json.dumps(compare(args.dataset, args.threshold), indent=2, sort_keys=True))
