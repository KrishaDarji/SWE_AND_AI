"""Statistical certification of an improvement between two configurations."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Certification:
    baseline_n: int
    candidate_n: int
    baseline_mean: float
    candidate_mean: float
    improvement: float
    t_statistic: float
    p_value: float
    confidence_low: float
    confidence_high: float
    credible: bool


def _number(value, row, column):
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"row {row}: {column!r} is not numeric") from exc


def _variance(values, mean):
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def certify_improvement(path: str, baseline: Mapping[str, object],
                        candidate: Mapping[str, object], measure: str = "",
                        alpha: float = 0.05, higher_is_better: bool = False) -> Certification:
    """Compare exact configuration rows with Welch's two-sample test.

    ``improvement`` is positive when the candidate is better. The p-value is
    two-sided and the confidence interval is for that improvement.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV must contain a header")
        headers = list(reader.fieldnames)
        target = measure or headers[-1]
        if target not in headers:
            raise ValueError(f"unknown measure column: {target}")
        settings = [name for name in headers if name != target]
        for label, config in (("baseline", baseline), ("candidate", candidate)):
            missing = [name for name in settings if name not in config]
            if missing:
                raise ValueError(f"{label} missing setting(s): {', '.join(missing)}")
        groups = {"baseline": [], "candidate": []}
        for row_number, row in enumerate(reader, 2):
            matches = []
            for label, config in (("baseline", baseline), ("candidate", candidate)):
                if all(str(row[name]).strip() == str(config[name]).strip() for name in settings):
                    groups[label].append(_number(row[target], row_number, target))
        if not groups["baseline"] or not groups["candidate"]:
            raise ValueError("both configurations must match at least one data row")
    b, c = groups["baseline"], groups["candidate"]
    bm, cm = sum(b) / len(b), sum(c) / len(c)
    direction = 1.0 if higher_is_better else -1.0
    improvement = direction * (cm - bm)
    if len(b) < 2 or len(c) < 2:
        return Certification(len(b), len(c), bm, cm, improvement, float("nan"),
                             float("nan"), float("nan"), float("nan"), False)
    se2 = _variance(b, bm) / len(b) + _variance(c, cm) / len(c)
    if se2 == 0:
        p_value = 0.0 if improvement != 0 else 1.0
        half = 0.0
        statistic = math.copysign(float("inf"), improvement) if improvement else 0.0
    else:
        statistic = improvement / math.sqrt(se2)
        p_value = _normal_two_sided_p(statistic)
        half = 1.96 * math.sqrt(se2)
    low, high = improvement - half, improvement + half
    return Certification(len(b), len(c), bm, cm, improvement, statistic, p_value,
                         low, high, improvement > 0 and p_value < alpha)
