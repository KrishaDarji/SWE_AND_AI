#!/usr/bin/env python3
"""Certify a pre-specified configuration-family contrast for Health-ClosedIssues."""
import csv
import math
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[8]
DATASET = REPO_ROOT / "moot/optimize/hpo/Health-ClosedIssues0000.csv"
FAMILY = "criterion"
TARGETS = {"MRE-": -1.0, "ACC+": 1.0, "PRED40+": 1.0}
GROUP_A, GROUP_B = "absolute_error", "poisson"
SEED, N_PERM, N_BOOT = 20261029, 10000, 5000


def load_validate(path=DATASET):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = set(TARGETS) | {FAMILY}
    if not rows or not required <= set(rows[0]):
        raise ValueError("missing required columns or empty dataset")
    for row in rows:
        for col in TARGETS:
            row[col] = float(row[col])
        if row[FAMILY] == "":
            raise ValueError("empty family value")
    return rows


def quality_scores(rows):
    bounds = {}
    for target, direction in TARGETS.items():
        values = [r[target] for r in rows]
        lo, hi = min(values), max(values)
        if hi == lo:
            raise ValueError(f"constant target: {target}")
        bounds[target] = (lo, hi, direction)
    scores = []
    for row in rows:
        components = []
        for target, (lo, hi, direction) in bounds.items():
            normalized = (row[target] - lo) / (hi - lo)
            components.append(normalized if direction > 0 else 1.0 - normalized)
        scores.append(sum(components) / len(components))
    return scores


def mean(values):
    return sum(values) / len(values)


def difference(a, b):
    return mean(b) - mean(a)  # B minus A; positive favors poisson


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def permutation_p(a, b, rng, n=N_PERM):
    observed = difference(a, b)
    pooled = a + b
    na = len(a)
    exceed = 0
    for _ in range(n):
        shuffled = list(pooled)
        rng.shuffle(shuffled)
        if difference(shuffled[:na], shuffled[na:]) >= observed:
            exceed += 1
    return (exceed + 1) / (n + 1)


def bootstrap_ci(a, b, rng, n=N_BOOT):
    effects = []
    for _ in range(n):
        aa = [a[rng.randrange(len(a))] for _ in a]
        bb = [b[rng.randrange(len(b))] for _ in b]
        effects.append(difference(aa, bb))
    return percentile(effects, .025), percentile(effects, .975)


def run(path=DATASET):
    rows = load_validate(path)
    scores = quality_scores(rows)
    groups = {name: [score for row, score in zip(rows, scores) if row[FAMILY] == name]
              for name in sorted({r[FAMILY] for r in rows})}
    if GROUP_A not in groups or GROUP_B not in groups:
        raise ValueError("primary contrast families are absent")
    rng = random.Random(SEED)
    a, b = groups[GROUP_A], groups[GROUP_B]
    effect = difference(a, b)
    ci = bootstrap_ci(a, b, rng)
    p = permutation_p(a, b, rng)
    credible = p < .05 and ci[0] > 0
    lines = [
        "Health-ClosedIssues certification",
        f"rows={len(rows)} targets={','.join(TARGETS)} family={FAMILY}",
        "quality_score=mean(min-max normalized targets; MRE reversed; ACC/PRED positive)",
        f"families={'; '.join(f'{k}: n={len(v)}, mean={mean(v):.6f}' for k,v in groups.items())}",
        f"contrast={GROUP_B} minus {GROUP_A}: effect={effect:.6f} bootstrap_95ci=[{ci[0]:.6f}, {ci[1]:.6f}] permutation_p_one_sided={p:.6f}",
        f"statistically_credible_at_alpha_0.05={'yes' if credible else 'no'}",
        "target_means=" + "; ".join(f"{t}: {GROUP_A}={mean([r[t] for r in rows if r[FAMILY]==GROUP_A]):.6f}, {GROUP_B}={mean([r[t] for r in rows if r[FAMILY]==GROUP_B]):.6f}" for t in TARGETS),
        "interpretation=The observed association supports a quality difference only under this dataset and contrast; it does not prove criterion causes it.",
        "limitations=Rows may not be independent; min-max scaling is sample-dependent; multiple targets/families and unmeasured configuration interactions can affect results; permutation exchangeability is an assumption.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        sys.stdout.write(run(Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
