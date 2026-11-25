#!/usr/bin/env python3
"""Independent certification analysis for nasa93dem."""
import csv
import math
import random
import statistics
from pathlib import Path

from sklearn.utils import check_random_state

TARGET = "Effort-"
GROUP = "centerX"
GROUP_A = "2"
GROUP_B = "5"
SEED = 20261029
PERMUTATIONS = 20000
BOOTSTRAPS = 10000


def load_validate(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("dataset is empty")
    required = {TARGET, GROUP}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    for row in rows:
        if not row[TARGET].strip() or not row[GROUP].strip():
            raise ValueError("missing target or group value")
        try:
            row["_effort"] = float(row[TARGET].strip())
        except ValueError as exc:
            raise ValueError("Effort- must be numeric") from exc
        if not math.isfinite(row["_effort"]):
            raise ValueError("Effort- must be finite")
    return rows


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def analyze(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row[GROUP].strip(), []).append(row["_effort"])
    eligible = {key: value for key, value in groups.items() if len(value) >= 5}
    if GROUP_A not in eligible or GROUP_B not in eligible:
        raise ValueError("predeclared comparison groups are unavailable")
    a, b = eligible[GROUP_A], eligible[GROUP_B]
    observed = statistics.mean(b) - statistics.mean(a)  # positive means A is lower
    pooled = math.sqrt(((len(a)-1)*statistics.pvariance(a) + (len(b)-1)*statistics.pvariance(b)) / (len(a)+len(b)-2))
    effect = observed / pooled if pooled else 0.0

    rng = check_random_state(SEED)
    boot = []
    for _ in range(BOOTSTRAPS):
        sa = [a[i] for i in rng.randint(0, len(a), len(a))]
        sb = [b[i] for i in rng.randint(0, len(b), len(b))]
        boot.append(statistics.mean(sb) - statistics.mean(sa))
    ci = (percentile(boot, .025), percentile(boot, .975))

    combined = a + b
    count = 0
    for _ in range(PERMUTATIONS):
        perm = rng.permutation(combined)
        statistic = statistics.mean(perm[len(a):]) - statistics.mean(perm[:len(a)])
        if statistic >= observed:
            count += 1
    p_value = (count + 1) / (PERMUTATIONS + 1)
    return {"n_total": len(rows), "groups": groups, "a": a, "b": b,
            "observed": observed, "effect": effect, "ci": ci,
            "p_value": p_value, "credible": p_value < .05 and ci[0] > 0}


def format_report(result):
    a, b = result["a"], result["b"]
    lo, hi = result["ci"]
    verdict = "YES" if result["credible"] else "NO"
    return f"""NASA93DEM certify analysis
Dataset rows: {result['n_total']}
Target: {TARGET}; process-group feature: {GROUP}
Comparison: centerX={GROUP_A} (A) vs centerX={GROUP_B} (B); positive difference means A has lower mean Effort-
Group sizes: A n={len(a)}, B n={len(b)}
Mean Effort-: A={statistics.mean(a):.6f}, B={statistics.mean(b):.6f}
Observed mean difference (B-A): {result['observed']:.6f}
Effect size (pooled standardized mean difference): {result['effect']:.6f}
95% bootstrap CI for (B-A): [{lo:.6f}, {hi:.6f}]
One-sided permutation p-value (A lower; {PERMUTATIONS} permutations, seed={SEED}): {result['p_value']:.6f}
Statistically credible at alpha=0.05: {verdict}
Interpretation: A lower mean is {'supported' if result['credible'] else 'not established'} by this resampling analysis.
Practical significance: the effect size and CI quantify magnitude in Effort- units; statistical significance alone does not establish a practically important difference.
Assumptions/uncertainty: rows are treated as independent; the groups are fixed before testing; bootstrap and permutation results are Monte Carlo estimates; skew, outliers, confounding, and small groups may affect the result.
Limitation: centerX is observational. This association is not proof that changing process center causes Effort- to change; no causal adjustment or experiment is provided.
"""


def main():
    path = Path(__file__).parents[8] / "moot" / "optimize" / "process" / "nasa93dem.csv"
    result = analyze(load_validate(path))
    print(format_report(result))


if __name__ == "__main__":
    main()
