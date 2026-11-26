"""The project's three-test ``same`` comparison from the ezr background.

Two samples are considered statistically indistinguishable only when all
three tests agree: pooled Cohen distance, Cliff's delta, and KS distance.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Sequence


def cohen(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        raise ValueError("both samples must be non-empty")
    nx, ny = len(xs), len(ys)
    pooled = math.sqrt(((nx - 1) * pstdev(xs) ** 2 + (ny - 1) * pstdev(ys) ** 2) / max(1, nx + ny - 2))
    return abs(mean(xs) - mean(ys)) / (pooled + 1e-32)


def cliffs(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        raise ValueError("both samples must be non-empty")
    greater = sum(x > y for x in xs for y in ys)
    less = sum(x < y for x in xs for y in ys)
    return abs(greater - less) / (len(xs) * len(ys))


def ks(xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs or not ys:
        raise ValueError("both samples must be non-empty")
    left, right = sorted(xs), sorted(ys)
    values = sorted(set(left + right))
    # ezr reports the empirical CDF gap in critical-value units, not the
    # unscaled D statistic.  Its decision threshold is therefore 1.36.
    d = max(abs(sum(x <= value for x in left) / len(left) - sum(y <= value for y in right) / len(right)) for value in values)
    return d / math.sqrt((len(left) + len(right)) / (len(left) * len(right)))


def same(xs: Sequence[float], ys: Sequence[float], *, cohen_limit: float = 0.35, cliffs_limit: float = 0.195, ks_limit: float = 1.36) -> dict[str, float | bool]:
    """Return component statistics and the authoritative all-three decision."""
    values = {"cohen": cohen(xs, ys), "cliffs": cliffs(xs, ys), "ks": ks(xs, ys)}
    values["same"] = values["cohen"] <= cohen_limit and values["cliffs"] <= cliffs_limit and values["ks"] <= ks_limit
    return values
